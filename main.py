# -*- coding: utf-8 -*-

from __future__ import annotations

"""Графическая утилита для сжатия изображений в WEBP.

Основные возможности:
* Тёмная тема с дашбордом нагрузки системы и прогресс-барами.
* Кнопки Старт, Стоп и Отмена (сброс состояния).
* Окно логов с чёрным фоном и зелёным текстом, отображающее ход работы.
* Поддержка исходников JPG/PNG/HEIC и др., сохранение структуры каталогов.
* Готово к сборке в исполняемый файл (единая точка входа, минимальные зависимости).
"""

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - опциональная зависимость для дашборда
    psutil = None

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - критичная зависимость
    raise SystemExit(
        "Для работы требуется Pillow (pip install pillow)."
    ) from exc

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - опциональная поддержка HEIC
    register_heif_opener = None

# Регистрируем поддержку HEIC/HEIF для Pillow
if register_heif_opener:
    register_heif_opener()

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 10 совместимость
    RESAMPLE_LANCZOS = Image.LANCZOS

# Целевой размер (может быть изменён из GUI)
TARGET_SIZE_KB = 300
TARGET_SIZE_BYTES = TARGET_SIZE_KB * 1024

# Качество, по которому перебираем
QUALITY_LEVELS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]

# Параметры изменения размера
SCALE_STEP = 0.8  # каждый шаг уменьшаем сторону до 80% от предыдущей
MIN_LONG_SIDE = 1200  # ниже этого по длинной стороне уже не ужимаем размером, дальше только качеством
MAX_START_LONG_SIDE = 2560  # если исходник длиннее этого — сначала ужимаем до этого значения


def set_compression_params(target_size_kb: int, min_long_side: int, max_start_long_side: int) -> None:
    """Обновить глобальные параметры компрессии из настроек GUI."""

    global TARGET_SIZE_KB, TARGET_SIZE_BYTES, MIN_LONG_SIDE, MAX_START_LONG_SIDE

    TARGET_SIZE_KB = target_size_kb
    TARGET_SIZE_BYTES = TARGET_SIZE_KB * 1024
    MIN_LONG_SIDE = min_long_side
    MAX_START_LONG_SIDE = max_start_long_side


def is_image_file(path: Path) -> bool:
    """Проверяем по расширению, что это поддерживаемое изображение."""

    base_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".bmp",
        ".tiff",
        ".tif",
        ".gif",
    }

    if register_heif_opener is None:
        base_exts -= {".heic", ".heif"}

    return path.suffix.lower() in base_exts


def open_image_any_format(input_path: Path) -> Image.Image:
    """Открывает изображение разных форматов."""

    return Image.open(input_path)


def _check_stop(stop_event: threading.Event | None):
    if stop_event and stop_event.is_set():
        raise KeyboardInterrupt


def compress_image_to_webp(
    input_path: Path,
    output_path: Path,
    log,
    stop_event: threading.Event | None = None,
) -> str:
    """
    Сжать один файл в WEBP с подробным логом.
    Показывает все шаги: загрузка, подбор качества, resize, проверка размера и финальный результат.
    Дополнительно:
    - если исходное фото существенно больше Full HD, сразу уменьшаем до MAX_START_LONG_SIDE по длинной стороне.
    """

    try:
        if output_path.exists():
            return f"[SKIP] {input_path.name} -> {output_path.name} (exists)"

        log(f"   - Загружаю изображение: {input_path.name}")
        base_img = open_image_any_format(input_path)
        _check_stop(stop_event)

        if base_img.mode not in ("RGB", "RGBA"):
            log("   - Привожу изображение к RGB")
            base_img = base_img.convert("RGB")

        has_alpha = base_img.mode == "RGBA"

        base_w, base_h = base_img.size
        log(f"   - Размер исходного файла: {base_w}x{base_h}")

        long_side = max(base_w, base_h)
        if long_side > MAX_START_LONG_SIDE:
            scale0 = MAX_START_LONG_SIDE / long_side
            new_w = max(1, int(base_w * scale0))
            new_h = max(1, int(base_h * scale0))
            log(
                "   - Изображение слишком большое (long_side=%s), "
                "сразу уменьшаю до %sx%s" % (long_side, new_w, new_h)
            )
            base_img = base_img.resize((new_w, new_h), RESAMPLE_LANCZOS)
            base_w, base_h = base_img.size
            log(f"   - Новый базовый размер: {base_w}x{base_h}")
        else:
            log("   - Размер в пределах лимита, работаю с оригиналом")

        scale = 1.0
        iteration = 0

        final_bytes = None
        final_quality = None
        final_size = None
        final_size_px = (base_w, base_h)

        while True:
            _check_stop(stop_event)
            iteration += 1
            log(f"   - Итерация #{iteration}")

            if scale < 1.0:
                new_w = max(1, int(base_w * scale))
                new_h = max(1, int(base_h * scale))
                log(f"     · Уменьшаю размер: {new_w}x{new_h}")
                img = base_img.resize((new_w, new_h), RESAMPLE_LANCZOS)
            else:
                img = base_img
                log(f"     · Использую базовый размер: {base_w}x{base_h}")

            log("     · Подбираю качество...")
            best_bytes = None
            best_quality = None

            for q in QUALITY_LEVELS:
                _check_stop(stop_event)
                buffer = BytesIO()
                save_kwargs = {
                    "format": "WEBP",
                    "quality": q,
                    "method": 6,
                }
                if has_alpha:
                    save_kwargs["lossless"] = False

                img.save(buffer, **save_kwargs)
                data = buffer.getvalue()
                size = len(data)

                size_kb = size // 1024
                log(f"       q={q}: {size_kb}KB")

                if size <= TARGET_SIZE_BYTES:
                    log(f"       ✓ Вписалось! ({size_kb}KB <= {TARGET_SIZE_KB}KB)")
                    best_bytes = data
                    best_quality = q
                    break

                if best_bytes is None or size < len(best_bytes):
                    best_bytes = data
                    best_quality = q

            final_bytes = best_bytes
            final_quality = best_quality
            final_size = len(best_bytes)
            final_size_px = img.size

            long_side_current = max(img.size)

            log(
                f"     · Лучший результат: Q={final_quality}, "
                f"{final_size // 1024}KB, {final_size_px[0]}x{final_size_px[1]}"
            )

            if final_size <= TARGET_SIZE_BYTES:
                log("     ✓ Файл вписался в лимит — завершаю подбор.\n")
                break

            if long_side_current <= MIN_LONG_SIDE:
                log("     ! Достигнут минимальный размер — дальше качество только ухудшится.\n")
                break

            scale *= SCALE_STEP
            log(f"     ↘ Размер > {TARGET_SIZE_KB}KB, уменьшаю масштаб до {round(scale, 3)}\n")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

        with open(tmp_output_path, "wb") as f:
            f.write(final_bytes)

        tmp_output_path.replace(output_path)

        return (
            f"[OK] {input_path.name} -> {output_path.name}  "
            f"(Q={final_quality}, {final_size // 1024}KB, "
            f"{final_size_px[0]}x{final_size_px[1]})"
        )

    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 - важно вернуть ошибку наверх
        return f"[ERROR] {input_path}: {e}"


def collect_files(input_dir: Path):
    """Собрать все поддерживаемые файлы рекурсивно."""

    return [p for p in input_dir.rglob("*") if p.is_file() and is_image_file(p)]


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{h} ч {m} мин {s} сек"
    elif m > 0:
        return f"{m} мин {s} сек"
    else:
        return f"{s} сек"


def make_progress_bar(progress: float, length: int = 30) -> str:
    """Текстовый прогресс-бар."""

    progress = max(0.0, min(1.0, progress))
    filled = int(length * progress)
    bar = "#" * filled + "-" * (length - filled)
    return f"[{bar}]"


def process_folder(
    input_dir: Path,
    output_dir: Path,
    on_progress=None,
    on_message=None,
    stop_event: threading.Event | None = None,
):
    """Основной цикл обработки для консоли и GUI."""

    log = on_message or (lambda msg: print(msg, flush=True))

    all_files = collect_files(input_dir)
    total_found = len(all_files)

    if total_found == 0:
        log("Нет изображений для обработки.")
        return {
            "done": 0,
            "total": 0,
            "interrupted": False,
            "elapsed": 0.0,
        }

    todo = []
    already_done = 0
    for src in all_files:
        rel = src.relative_to(input_dir)
        dst = (output_dir / rel).with_suffix(".webp")
        if dst.exists():
            already_done += 1
        else:
            todo.append((src, dst))

    total_todo = len(todo)

    log(f"Всего найдено файлов:          {total_found}")
    log(f"Уже обработано ранее:          {already_done}")
    log(f"Осталось обработать в этот раз: {total_todo}")
    log(f"Входная папка:   {input_dir}")
    log(f"Выходная папка:  {output_dir}")
    log("Структура подпапок будет сохранена.\n")

    if total_todo == 0:
        log("Все файлы уже обработаны, делать больше нечего.")
        return {
            "done": 0,
            "total": total_todo,
            "interrupted": False,
            "elapsed": 0.0,
        }

    start_time = time.time()
    done = 0
    interrupted = False

    try:
        for src, dst in todo:
            _check_stop(stop_event)
            msg = compress_image_to_webp(src, dst, log, stop_event=stop_event)
            done += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / done
            remaining = (total_todo - done) * avg_time
            progress = done / total_todo

            bar = make_progress_bar(progress)
            eta_str = format_time(remaining)

            log(f"{done}/{total_todo} {bar} | ETA: ~{eta_str} | {msg}")
            if on_progress:
                on_progress(done, total_todo, progress, eta_str)

    except KeyboardInterrupt:
        interrupted = True
        log("\n\nОстановка по запросу пользователя (Стоп/Cancel).")
        log("Уже готовые .webp-файлы сохранены и будут пропущены при следующем запуске.")

    total_time = time.time() - start_time
    log("\nИтог:")
    log(f"Обработано в этом запуске: {done} из {total_todo}")
    log(f"Затраченное время:         {format_time(total_time)}")

    if interrupted:
        log("При следующем запуске с теми же параметрами скрипт продолжит с оставшихся файлов.\n")
    else:
        log("Все запланированные файлы обработаны.\n")

    return {
        "done": done,
        "total": total_todo,
        "interrupted": interrupted,
        "elapsed": total_time,
    }


class CompressorGUI:
    """Тёмная графическая оболочка с дашбордом и логами."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Compressor - WEBP")
        self.root.configure(bg="#0f1115")
        self.root.geometry("980x720")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_event: threading.Event | None = None
        self.worker: threading.Thread | None = None
        self.is_running = False

        self.target_size_var = tk.StringVar(value=str(TARGET_SIZE_KB))
        self.min_long_side_var = tk.StringVar(value=str(MIN_LONG_SIDE))
        self.max_start_long_side_var = tk.StringVar(value=str(MAX_START_LONG_SIDE))

        self._build_styles()
        self._build_layout()
        self._schedule_log_update()
        self._schedule_system_stats()

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        dark_bg = "#0f1115"
        card_bg = "#161925"
        accent = "#3fb27f"
        text_color = "#e0e6ed"

        style.configure(
            "TFrame",
            background=dark_bg,
        )
        style.configure(
            "Card.TFrame",
            background=card_bg,
            relief="groove",
        )
        style.configure(
            "TLabel",
            background=dark_bg,
            foreground=text_color,
        )
        style.configure(
            "Card.TLabel",
            background=card_bg,
            foreground=text_color,
            padding=4,
        )
        style.configure(
            "Accent.TButton",
            background=accent,
            foreground="#0f1115",
            padding=10,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#48c58a")],
        )
        style.configure(
            "Danger.TButton",
            background="#c65353",
            foreground="#0f1115",
            padding=10,
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#d96a6a")],
        )
        style.configure(
            "Secondary.TButton",
            background="#2b3140",
            foreground=text_color,
            padding=10,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#343c4f")],
        )
        style.configure(
            "TProgressbar",
            troughcolor="#1c2230",
            background=accent,
            bordercolor="#1c2230",
            lightcolor=accent,
            darkcolor=accent,
        )

    def _build_layout(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=16, pady=12)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ожидание запуска")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="0 / 0")
        self.eta_var = tk.StringVar(value="ETA: --")

        self._build_path_selector(top_frame)
        self._build_parameters(top_frame)
        self._build_controls(top_frame)

        dashboard = ttk.Frame(self.root, style="Card.TFrame")
        dashboard.pack(fill=tk.X, padx=16, pady=(0, 12))
        self._build_dashboard(dashboard)

        progress_frame = ttk.Frame(self.root, style="Card.TFrame")
        progress_frame.pack(fill=tk.X, padx=16, pady=(0, 12))
        self._build_progress(progress_frame)

        log_frame = ttk.Frame(self.root, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        self._build_log(log_frame)

    def _build_path_selector(self, parent):
        paths = ttk.Frame(parent)
        paths.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(paths, text="Входная папка").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(paths, textvariable=self.input_var, width=70)
        input_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(paths, text="Обзор", command=self._choose_input, style="Secondary.TButton").grid(
            row=1, column=1, padx=(0, 8)
        )

        ttk.Label(paths, text="Выходная папка").grid(row=2, column=0, sticky="w", pady=(8, 0))
        output_entry = ttk.Entry(paths, textvariable=self.output_var, width=70)
        output_entry.grid(row=3, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(paths, text="Обзор", command=self._choose_output, style="Secondary.TButton").grid(
            row=3, column=1, padx=(0, 8)
        )

        paths.columnconfigure(0, weight=1)
        input_entry.configure(foreground="#e0e6ed", background="#1a1f2b")
        output_entry.configure(foreground="#e0e6ed", background="#1a1f2b")

    def _build_parameters(self, parent):
        params = ttk.Frame(parent)
        params.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(params, text="Настройки сжатия (редактируемые)", font=("Segoe UI", 11, "bold"))\
            .grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        ttk.Label(params, text="Целевой размер, KB").grid(row=1, column=0, sticky="w")
        target_entry = ttk.Entry(params, textvariable=self.target_size_var, width=10)
        target_entry.grid(row=1, column=1, padx=(6, 18))

        ttk.Label(params, text="Мин. длинная сторона, px").grid(row=1, column=2, sticky="w")
        min_entry = ttk.Entry(params, textvariable=self.min_long_side_var, width=10)
        min_entry.grid(row=1, column=3, padx=(6, 18))

        ttk.Label(params, text="Ограничение исходника, px").grid(row=1, column=4, sticky="w")
        max_entry = ttk.Entry(params, textvariable=self.max_start_long_side_var, width=10)
        max_entry.grid(row=1, column=5, padx=(6, 0))

        for entry in (target_entry, min_entry, max_entry):
            entry.configure(foreground="#e0e6ed", background="#1a1f2b")

        for col in range(6):
            params.columnconfigure(col, weight=1)

    def _build_controls(self, parent):
        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X)

        ttk.Label(buttons, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).pack(
            side=tk.LEFT, padx=(0, 12)
        )

        self.start_btn = ttk.Button(buttons, text="Старт", command=self.start, style="Accent.TButton")
        self.start_btn.pack(side=tk.RIGHT, padx=(8, 0))

        self.stop_btn = ttk.Button(buttons, text="Стоп", command=self.stop, style="Danger.TButton")
        self.stop_btn.pack(side=tk.RIGHT, padx=(8, 0))
        self.stop_btn.state(["disabled"])

        self.reset_btn = ttk.Button(buttons, text="Отмена", command=self.reset, style="Secondary.TButton")
        self.reset_btn.pack(side=tk.RIGHT)

    def _build_dashboard(self, parent):
        parent.columnconfigure((0, 1, 2), weight=1)

        self.cpu_var = tk.StringVar(value="CPU: --")
        self.mem_var = tk.StringVar(value="RAM: --")
        self.files_var = tk.StringVar(value="Файлы: 0/0")

        ttk.Label(parent, textvariable=self.cpu_var, style="Card.TLabel", font=("Segoe UI", 11)).grid(
            row=0, column=0, sticky="w", padx=12, pady=10
        )
        ttk.Label(parent, textvariable=self.mem_var, style="Card.TLabel", font=("Segoe UI", 11)).grid(
            row=0, column=1, sticky="w", padx=12, pady=10
        )
        ttk.Label(parent, textvariable=self.files_var, style="Card.TLabel", font=("Segoe UI", 11)).grid(
            row=0, column=2, sticky="w", padx=12, pady=10
        )

    def _build_progress(self, parent):
        ttk.Label(parent, text="Прогресс", style="Card.TLabel", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=12, pady=(12, 6)
        )

        self.progressbar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progressbar.pack(fill=tk.X, padx=12, pady=(0, 6))

        info_frame = ttk.Frame(parent, style="Card.TFrame")
        info_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Label(info_frame, textvariable=self.progress_text_var, style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(info_frame, textvariable=self.eta_var, style="Card.TLabel").pack(side=tk.RIGHT)

    def _build_log(self, parent):
        ttk.Label(parent, text="Логи работы", style="Card.TLabel", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", padx=12, pady=(12, 6)
        )
        self.log_text = tk.Text(
            parent,
            bg="#000000",
            fg="#39ff14",
            insertbackground="#39ff14",
            font=("Consolas", 10),
            height=18,
            wrap="word",
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.log_text.configure(state="disabled")

        scrollbar = ttk.Scrollbar(parent, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text["yscrollcommand"] = scrollbar.set

    def _choose_input(self):
        folder = filedialog.askdirectory(title="Выберите входную папку")
        if folder:
            self.input_var.set(folder)
            default_output = str(Path(folder).resolve().parent / "compressed_webp")
            if not self.output_var.get():
                self.output_var.set(default_output)

    def _choose_output(self):
        folder = filedialog.askdirectory(title="Выберите выходную папку")
        if folder:
            self.output_var.set(folder)

    def _log(self, message: str):
        self.log_queue.put(message)

    def _schedule_log_update(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._schedule_log_update)

    def _schedule_system_stats(self):
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self.cpu_var.set(f"CPU: {cpu:.0f}%")
            self.mem_var.set(f"RAM: {mem:.0f}%")
        else:
            self.cpu_var.set("CPU: N/A")
            self.mem_var.set("RAM: N/A")
        self.root.after(1000, self._schedule_system_stats)

    def start(self):
        if self.is_running:
            return

        input_path = Path(self.input_var.get()).expanduser()
        output_path = Path(self.output_var.get()).expanduser()

        parsed = self._parse_params()
        if parsed is None:
            return

        target_kb, min_long, max_long = parsed

        if not input_path.is_dir():
            messagebox.showerror("Ошибка", "Укажите корректную входную папку.")
            return

        if not output_path:
            messagebox.showerror("Ошибка", "Укажите выходную папку.")
            return

        self.progress_var.set(0)
        self.progress_text_var.set("0 / 0")
        self.eta_var.set("ETA: --")
        self.files_var.set("Файлы: 0/0")
        self.status_var.set("Запуск...")
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

        set_compression_params(target_kb, min_long, max_long)

        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self._run_compression,
            args=(input_path, output_path),
            daemon=True,
        )
        self.is_running = True
        self.worker.start()
        self.start_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self.status_var.set("В работе")

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.status_var.set("Остановка...")

    def reset(self):
        if self.is_running and self.stop_event:
            self.stop_event.set()

        self.is_running = False
        self.progress_var.set(0)
        self.progress_text_var.set("0 / 0")
        self.eta_var.set("ETA: --")
        self.files_var.set("Файлы: 0/0")
        self.status_var.set("Ожидание запуска")
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])

        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")

    def _parse_params(self):
        try:
            target_kb = int(self.target_size_var.get())
            min_long = int(self.min_long_side_var.get())
            max_long = int(self.max_start_long_side_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Параметры должны быть целыми числами.")
            return None

        if target_kb <= 0 or min_long <= 0 or max_long <= 0:
            messagebox.showerror("Ошибка", "Все параметры должны быть больше нуля.")
            return None

        if max_long < min_long:
            messagebox.showerror(
                "Ошибка",
                "Ограничение исходника должно быть не меньше минимальной длинной стороны.",
            )
            return None

        return target_kb, min_long, max_long

    def _on_progress(self, done: int, total: int, progress: float, eta: str):
        self.progress_var.set(progress * 100)
        self.progress_text_var.set(f"{done} / {total}")
        self.eta_var.set(f"ETA: ~{eta}")
        self.files_var.set(f"Файлы: {done}/{total}")

    def _run_compression(self, input_path: Path, output_path: Path):
        try:
            process_folder(
                input_dir=input_path,
                output_dir=output_path,
                on_progress=self._on_progress,
                on_message=self._log,
                stop_event=self.stop_event,
            )
        finally:
            self.root.after(0, self._finish_run)

    def _finish_run(self):
        self.is_running = False
        self.start_btn.state(["!disabled"])
        self.stop_btn.state(["disabled"])
        self.status_var.set("Готово" if not (self.stop_event and self.stop_event.is_set()) else "Остановлено")


def run_cli():
    parser = argparse.ArgumentParser(
        description="Сжать все изображения в папке до WEBP ~300KB (однопоточно, с прогресс-баром, ETA, возобновлением и адаптивным размером).",
    )
    parser.add_argument(
        "input_folder",
        help="Папка с исходными фотографиями (например, photos)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        help="Папка для сохранения (по умолчанию: compressed_webp рядом с входной)",
        default=None,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_folder).resolve()
    if not input_dir.is_dir():
        print("Указанная папка не существует или это не папка.")
        raise SystemExit(1)

    if args.output_folder:
        output_dir = Path(args.output_folder).resolve()
    else:
        output_dir = input_dir.parent / "compressed_webp"

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
    )


def run_gui():
    root = tk.Tk()
    CompressorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        run_cli()
    else:
        run_gui()
