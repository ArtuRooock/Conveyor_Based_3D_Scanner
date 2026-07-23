import cv2
import os
import sys

numCam = int(input("Enter number of camera: "))
directory = f"Frames/Cam{numCam}_Step2"
os.makedirs(directory, exist_ok=True)

camera_index = 1
cap = cv2.VideoCapture(camera_index)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print(f"Не удалось открыть камеру с индексом {camera_index}")
    sys.exit(1)

print(f"Размер кадра: {width}x{height}")
print("Нажмите ПРОБЕЛ, чтобы сохранить текущий кадр.")
print("Нажмите 'q' или ESC для выхода.")

# Создаём окно и устанавливаем его размер под кадр (чтобы не масштабировалось)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", width, height)

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        filename = f"china{numCam}_test.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        print(f"Изображение сохранено: {filepath} (размер {width}x{height})")
        frame_counter += 1
    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()