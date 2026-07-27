import cv2

cap = cv2.VideoCapture(0)  # для Windows часто помогает DShow
numCam = int(input("Number of camera: "))
numStep = int(input("Number of step: "))

# print(cap.get(cv2.CAP_PROP_FOURCC))

# Устанавливаем желаемое разрешение
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# Получаем реальное разрешение (камера могла не поддержать запрошенное)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Реальное разрешение: {width}x{height}")

# Если размеры не совпадают с ожидаемыми — выводим предупреждение
if width != 1280 or height != 800:
    print("Внимание: камера не поддерживает 1280x800, используется", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print(f'InputVideos/Cam{numCam}/Cam{numCam}_{numStep}.avi')
out = cv2.VideoWriter(f'InputVideos/Cam{numCam}/Cam{numCam}_{numStep}.avi', fourcc, 120.0, (width, height))
if not out.isOpened():
    raise RuntimeError("VideoWriter fail")
# Частота 30 FPS — более реалистична для MJPG на 1280x800, чем 120

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_for_record = frame.copy()
    frame[height // 2 : height // 2 + 3, :] = 0
    out.write(frame_for_record)
    cv2.imshow('Recording...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()