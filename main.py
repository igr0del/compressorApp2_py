# -*- coding: utf-8 -*-

import time
from io import BytesIO
from pathlib import Path

import rawpy
from PIL import Image
from pillow_heif import register_heif_opener

# Регистрируем поддержку HEIC/HEIF для Pillow
register_heif_opener()

# Целевой размер
TARGET_SIZE_KB = 300
TARGET_SIZE_BYTES = TARGET_SIZE_KB * 1024

# Качество, по которому перебираем
QUALITY_LEVELS = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]

# Параметры изменения размера
SCALE_STEP = 0.8           # каждый шаг уменьшаем сторону до 80% от предыдущей
MIN_LONG_SIDE = 1200       # ниже этого по длинной стороне уже не ужимаем размером, дальше только качеством
MAX_START_LONG_SIDE = 2560 # если исходник длиннее этого — сначала ужимаем до этого значения


def is_image_file(path: Path) -> bool:
    """Проверяем по расширению, что это поддерживаемое изображение."""
    return path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".heic", ".heif",
        ".bmp", ".tiff", ".tif", ".gif",
        ".dng", ".arw",  # RAW (DNG, Sony ARW)
    }


def open_image_any_format(input_path: Path) -> Image.Image:
    """
    Открывает изображение разных форматов.
    Для RAW (DNG, ARW) используем rawpy, для остальных — Pillow.
    """
    ext = input_path.suffix.lower()

    if ext in {".dng", ".arw"}:
        with rawpy.imread(str(input_path)) as raw:
            rgb = raw.postprocess()
        return Image.fromarray(rgb, "RGB")

    return Image.open(input_path)


def compress_image_to_webp(input_path: Path, output_path: Path) -> str:
    """
    Сжать один файл в WEBP с подробным логом.
    Показывает все шаги: загрузка, подбор качества, resize, проверка размера и финальный результат.
    Дополнительно:
    - если исходное фото существенно больше Full HD, сразу уменьшаем до MAX_START_LONG_SIDE по длинной стороне.
    """
    try:
        if output_path.exists():
            return f"[SKIP] {input_path.name} -> {output_path.name} (exists)"

        print(f"   - Загружаю изображение: {input_path.name}", flush=True)
        base_img = open_image_any_format(input_path)

        if base_img.mode not in ("RGB", "RGBA"):
            print(f"   - Привожу изображение к RGB", flush=True)
            base_img = base_img.convert("RGB")

        has_alpha = base_img.mode == "RGBA"

        base_w, base_h = base_img.size
        print(f"   - Размер исходного файла: {base_w}x{base_h}", flush=True)

        # --- Быстрый даунскейл очень больших изображений ---
        long_side = max(base_w, base_h)
        if long_side > MAX_START_LONG_SIDE:
            scale0 = MAX_START_LONG_SIDE / long_side
            new_w = max(1, int(base_w * scale0))
            new_h = max(1, int(base_h * scale0))
            print(
                f"   - Изображение слишком большое (long_side={long_side}), "
                f"сразу уменьшаю до {new_w}x{new_h}", flush=True
            )
            base_img = base_img.resize((new_w, new_h), Image.LANCZOS)
            base_w, base_h = base_img.size
            print(f"   - Новый базовый размер: {base_w}x{base_h}", flush=True)
        else:
            print("   - Размер в пределах лимита, работаю с оригиналом", flush=True)

        scale = 1.0
        iteration = 0

        final_bytes = None
        final_quality = None
        final_size = None
        final_size_px = (base_w, base_h)

        while True:
            iteration += 1
            print(f"   - Итерация #{iteration}", flush=True)

            # Масштабирование
            if scale < 1.0:
                new_w = max(1, int(base_w * scale))
                new_h = max(1, int(base_h * scale))
                print(f"     · Уменьшаю размер: {new_w}x{new_h}", flush=True)
                img = base_img.resize((new_w, new_h), Image.LANCZOS)
            else:
                img = base_img
                print(f"     · Использую базовый размер: {base_w}x{base_h}", flush=True)

            print(f"     · Подбираю качество...", flush=True)

            best_bytes = None
            best_quality = None

            # Подбор качества
            for q in QUALITY_LEVELS:
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
                print(f"       q={q}: {size_kb}KB", flush=True)

                if size <= TARGET_SIZE_BYTES:
                    print(f"       ✓ Вписалось! ({size_kb}KB <= {TARGET_SIZE_KB}KB)", flush=True)
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

            print(
                f"     · Лучший результат: Q={final_quality}, "
                f"{final_size // 1024}KB, {final_size_px[0]}x{final_size_px[1]}",
                flush=True,
            )

            # Условия выхода из цикла
            if final_size <= TARGET_SIZE_BYTES:
                print("     ✓ Файл вписался в лимит — завершаю подбор.\n", flush=True)
                break

            if long_side_current <= MIN_LONG_SIDE:
                print("     ! Достигнут минимальный размер — дальше качество только ухудшится.\n", flush=True)
                break

            # Уменьшаем разрешение и повторяем
            scale *= SCALE_STEP
            print(f"     ↘ Размер > {TARGET_SIZE_KB}KB, уменьшаю масштаб до {round(scale, 3)}\n", flush=True)

        # Сохранение
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

        with open(tmp_output_path, "wb") as f:
            f.write(final_bytes)

        tmp_output_path.replace(output_path)

        return (f"[OK] {input_path.name} -> {output_path.name}  "
                f"(Q={final_quality}, {final_size // 1024}KB, "
                f"{final_size_px[0]}x{final_size_px[1]})")

    except Exception as e:
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


def process_folder(input_dir: Path, output_dir: Path):
    # Все исходные картинки
    all_files = collect_files(input_dir)
    total_found = len(all_files)

    if total_found == 0:
        print("Нет изображений для обработки.")
        return

    # Фильтруем только те, для которых ещё нет .webp (для возобновления)
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

    print(f"Всего найдено файлов:          {total_found}")
    print(f"Уже обработано ранее:          {already_done}")
    print(f"Осталось обработать в этот раз: {total_todo}")
    print(f"Входная папка:   {input_dir}")
    print(f"Выходная папка:  {output_dir}")
    print("Структура подпапок будет сохранена.\n")

    if total_todo == 0:
        print("Все файлы уже обработаны, делать больше нечего.")
        return

    start_time = time.time()
    done = 0
    interrupted = False

    try:
        for src, dst in todo:
            msg = compress_image_to_webp(src, dst)
            done += 1

            elapsed = time.time() - start_time
            avg_time = elapsed / done
            remaining = (total_todo - done) * avg_time
            progress = done / total_todo

            bar = make_progress_bar(progress)
            eta_str = format_time(remaining)

            print(f"{done}/{total_todo} {bar} | ETA: ~{eta_str} | {msg}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\nОстановка по запросу пользователя (Ctrl+C).")
        print("Уже готовые .webp-файлы сохранены и будут пропущены при следующем запуске.")

    total_time = time.time() - start_time
    print("\nИтог:")
    print(f"Обработано в этом запуске: {done} из {total_todo}")
    print(f"Затраченное время:         {format_time(total_time)}")

    if interrupted:
        print("При следующем запуске с теми же параметрами скрипт продолжит с оставшихся файлов.\n")
    else:
        print("Все запланированные файлы обработаны.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Сжать все изображения в папке до WEBP ~300KB (однопоточно, с прогресс-баром, ETA, возобновлением и адаптивным размером)."
    )
    parser.add_argument(
        "input_folder",
        help="Папка с исходными фотографиями (например, photos)",
    )
    parser.add_argument(
        "-o", "--output-folder",
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
        # по умолчанию: ../compressed_webp
        output_dir = input_dir.parent / "compressed_webp"

    process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
    )
