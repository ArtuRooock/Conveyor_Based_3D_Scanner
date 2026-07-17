import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # для Windows часто помогает DShow

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
out = cv2.VideoWriter('InputVideos/Cam2/Cam2_3.avi', fourcc, 120.0, (width, height))
# Частота 30 FPS — более реалистична для MJPG на 1280x800, чем 120

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Recording...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()