# Виправляємо розмітку на комп'ютері (конвертуємо полігони в прямокутники),
# Завантаження нового правильного датасету та запуск тренування повністю вирішить проблему з кількістю кораблів і якістю детекції.

import os
import glob


def convert_segmentation_to_detection(labels_dir):
    converted_files = 0
    converted_lines = 0

    # Шукаємо всі .txt файли у папці з мітками (включно з підпапками train/valid/test)
    txt_files = glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    print(f"[INFO] Знайдено файлів розмітки: {len(txt_files)}")

    for file_path in txt_files:
        new_lines = []
        file_changed = False

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = parts[0]
            coords = [float(x) for x in parts[1:]]

            # Якщо елементів більше 5 — це полігон (сегментація)
            if len(coords) > 4:
                xs = coords[0::2]  # Непарні індекси (X)
                ys = coords[1::2]  # Парні індекси (Y)

                # Обчислюємо обмежувальний прямокутник (Bounding Box)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                width = x_max - x_min
                height = y_max - y_min

                # Захист від нульових розмірів
                if width <= 0 or height <= 0:
                    continue

                # Формуємо стандартний рядок детекції YOLO (клас, x_center, y_center, w, h)
                new_line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                new_lines.append(new_line)
                file_changed = True
                converted_lines += 1
            else:
                # Якщо рядок вже у форматі детекції (5 значень), залишаємо без змін
                new_lines.append(line if line.endswith('\n') else line + '\n')

        if file_changed:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            converted_files += 1

    print(f"[УСПІХ] Опрацьовано файлів зі змінами: {converted_files}")
    print(f"[УСПІХ] Конвертовано полігонів у прямокутники: {converted_lines}")


if __name__ == "__main__":
    # Вкажіть шлях до кореневої папки вашого датасету на комп'ютері
    DATASET_LABELS_PATH = r"D:\IT_STEP_Academy\EXAM\train\labels"

    convert_segmentation_to_detection(DATASET_LABELS_PATH)

# -------------------------------------------------------------
#  РЕЗУЛЬТАТ КОНВЕРТУВАННЯ ПОЛІГОНІВ У ПРЯМОКУТНИКИ
# -------------------------------------------------------------
# ШЛЯХ ДО ФАЙЛУ:
#           D:\IT_STEP_Academy\AI\ITStep-AI\.venv\Scripts\python.exe D:\IT_STEP_Academy\AI\ITStep-AI\convert_labels.py

# [INFO] Знайдено файлів розмітки: 2094
# [УСПІХ] Опрацьовано файлів зі змінами: 1394
# [УСПІХ] Конвертовано полігонів у прямокутники: 34879
#
# Process finished with exit code 0