import cv2
import serial
import pandas as pd
import os
import time

# --------------------------
# Настройки
# --------------------------
SERIAL_PORT = "COM5"          # Windows
# SERIAL_PORT = "/dev/ttyUSB0"   # Linux
BAUDRATE = 115200

# Идентификаторы камер (подберите под свои)
CAMERA_LEFT = 0
CAMERA_RIGHT = 1

# Папка для сохранения
SAVE_FOLDER = "dataset"
VIDEO_FOLDER = os.path.join(SAVE_FOLDER, "videos")
CSV_FILE = os.path.join(SAVE_FOLDER, "encoder_data.csv")

os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Настройки видео
FPS = 30.0                    # можно выставить реальное FPS камер
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')   # или 'XVID'

# Имена видеофайлов
VIDEO_LEFT = os.path.join(VIDEO_FOLDER, "video_left.mp4")
VIDEO_RIGHT = os.path.join(VIDEO_FOLDER, "video_right.mp4")

# --------------------------
# Подключение камер и порта
# --------------------------
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

cap_left = cv2.VideoCapture(CAMERA_LEFT)
cap_right = cv2.VideoCapture(CAMERA_RIGHT)

if not cap_left.isOpened():
    raise RuntimeError("Не удалось открыть левую камеру")
if not cap_right.isOpened():
    raise RuntimeError("Не удалось открыть правую камеру")

# Получаем размер кадра (предполагаем, что обе камеры одинакового разрешения)
width  = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаём VideoWriter'ы
out_left = cv2.VideoWriter(VIDEO_LEFT, FOURCC, FPS, (width, height))
out_right = cv2.VideoWriter(VIDEO_RIGHT, FOURCC, FPS, (width, height))

# Счётчики кадров для каждой камеры
frame_idx_left = 0
frame_idx_right = 0

# Список для CSV-строк
rows = []

# Переменная для хранения последних данных энкодера
last_encoder = None

print("Нажмите ESC для остановки")
print("Запись видео начата...")

while True:
    # ---- Чтение данных энкодера (всех накопившихся) ----
    encoder = None
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        try:
            position, time_us, rps, speed = line.split(",")
            encoder = {
                "position": int(position),
                "time_us": int(time_us),
                "rps": float(rps),
                "speed_mm_s": float(speed)
            }
        except:
            pass
    if encoder is not None:
        last_encoder = encoder

    # ---- Захват кадра с левой камеры ----
    ret_left, frame_left = cap_left.read()
    if ret_left:
        # Запись видео
        out_left.write(frame_left)
        # Сохранение данных в CSV
        timestamp = time.time()
        row = {
            "camera_id": 0,
            "frame_index": frame_idx_left,
            "timestamp": timestamp,
            "position": last_encoder["position"] if last_encoder else None,
            "time_us": last_encoder["time_us"] if last_encoder else None,
            "rps": last_encoder["rps"] if last_encoder else None,
            "speed_mm_s": last_encoder["speed_mm_s"] if last_encoder else None
        }
        rows.append(row)
        frame_idx_left += 1
        # Отображение (можно отдельно или объединить)
        cv2.imshow("Left", frame_left)

    # ---- Захват кадра с правой камеры ----
    ret_right, frame_right = cap_right.read()
    if ret_right:
        out_right.write(frame_right)
        timestamp = time.time()
        row = {
            "camera_id": 1,
            "frame_index": frame_idx_right,
            "timestamp": timestamp,
            "position": last_encoder["position"] if last_encoder else None,
            "time_us": last_encoder["time_us"] if last_encoder else None,
            "rps": last_encoder["rps"] if last_encoder else None,
            "speed_mm_s": last_encoder["speed_mm_s"] if last_encoder else None
        }
        rows.append(row)
        frame_idx_right += 1
        cv2.imshow("Right", frame_right)

    # Выход по ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# --------------------------
# Завершение
# --------------------------
cap_left.release()
cap_right.release()
out_left.release()
out_right.release()
ser.close()
cv2.destroyAllWindows()

# Сохраняем CSV
df = pd.DataFrame(rows)
df.to_csv(CSV_FILE, index=False)

print(f"Записано кадров: левая = {frame_idx_left}, правая = {frame_idx_right}")
print(f"Всего строк в CSV: {len(df)}")
print(f"Видео сохранены в {VIDEO_FOLDER}")