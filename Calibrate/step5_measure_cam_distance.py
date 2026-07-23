import cv2
import numpy as np
import serial
import json
import threading
import time
import sys

import cheker_params_little as cheker_params

CALIB_FILE_CAM1 = r'CalibrationFiles/Cam1/calib_china1.npz'
CALIB_FILE_CAM2 = r'CalibrationFiles/Cam2/calib_china2.npz'

# Индексы камер (обычно 0 и 1, но могут быть другими)
CAM1_INDEX = 1
CAM2_INDEX = 2

# Настройки UART
UART_PORT = 'COM3'
UART_BAUDRATE = 115200

mtx1_per = np.load(r"CalibrationFiles/Cam1/plane_china1.npz")
mtx2_per = np.load(r"CalibrationFiles/Cam2/plane_china2.npz")  # поменять путь
e1 = mtx1_per['e']
e2 = mtx2_per['e']
pc1 = mtx1_per['pc']
pc2 = mtx2_per['pc']

# Минимальный сдвиг энкодера (в шагах), при котором калибровочный расчёт
# коэффициента считается достоверным. Если сдвиг меньше — просим повторить.
MIN_CALIB_ENCODER_DELTA = 5

# ------------------ Глобальные переменные для энкодера ------------------
last_position = None
position_lock = threading.Lock()


def serial_reader(port, baudrate):
    global last_position
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"[UART] Подключено к {port} @ {baudrate} бод")
        while True:
            raw = ser.readline()
            if raw:
                try:
                    decoded = raw.decode('utf-8').strip()
                except UnicodeDecodeError:
                    try:
                        decoded = raw.decode('cp1251').strip()
                    except UnicodeDecodeError:
                        continue
                if decoded:
                    try:
                        data = json.loads(decoded)
                        if 'pos' in data:
                            with position_lock:
                                last_position = data['pos']
                    except json.JSONDecodeError:
                        pass
            time.sleep(0.001)
    except Exception as e:
        print(f"[UART] Ошибка: {e}")
        sys.exit(1)


def get_current_position():
    with position_lock:
        return last_position


def find_board_pose(img, mtx, dist):
    """
    Находит шахматную доску на изображении и возвращает координаты
    её центра в СИСТЕМЕ КООРДИНАТ КАМЕРЫ (а не доски), а также rvec/tvec/corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, cheker_params.CHECKERBOARD, None)
    if not ret:
        return None, None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    ret, rvec, tvec = cv2.solvePnP(cheker_params.corner_coordinates, corners2, mtx, dist)
    if not ret:
        return None, None, None, None

    # Центр доски в её собственной системе координат
    center_board_local = np.mean(cheker_params.corner_coordinates, axis=0)

    # Перевод центра доски в систему координат камеры: R * p + t
    rot, _ = cv2.Rodrigues(rvec)
    center_board_cam = rot @ center_board_local + tvec.flatten()

    return center_board_cam, rvec, tvec, corners2


def project_to_conveyor(center_board_cam, e, pc):
    """
    Переводит точку из системы координат камеры в систему координат
    конвейера (см. step4_plane_calib.py): v_conv = e^T * (v_cam - pc)
    Возвращает координату вдоль оси ex (индекс 0) — оси движения ленты.
    """
    v_conv = e.T @ (center_board_cam - pc)
    return v_conv, v_conv[0]


def draw_overlay(display, step, cur_pos_display):
    labels = {
        1: "Cam1: снимок A сделан",
        2: "Cam1: снимок B сделан (коэфф. посчитан)",
        3: "Cam2: снимок сделан",
    }
    y = 30
    for s in range(1, step + 1):
        if s in labels:
            cv2.putText(display, labels[s], (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += 25

    cv2.putText(display, f"Шаг: {step}/3", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 30
    cv2.putText(display, f"Энкодер: {cur_pos_display}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def main():
    # Загрузка калибровок
    try:
        calib1 = np.load(CALIB_FILE_CAM1)
        mtx1, dist1 = calib1['mtx'], calib1['dist']
        calib2 = np.load(CALIB_FILE_CAM2)
        mtx2, dist2 = calib2['mtx'], calib2['dist']
        print("[INFO] Калибровки загружены.")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Открытие камер
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    reader_thread = threading.Thread(target=serial_reader, args=(UART_PORT, UART_BAUDRATE), daemon=True)
    reader_thread.start()

    step = 0

    # Шаг 0: первый снимок Cam1 (точка A)
    pos1a_raw = None
    center_board1a = None
    img1a = None

    # Шаг 1: второй снимок Cam1 после сдвига конвейера (точка B) — для калибровки коэффициента
    pos1b_raw = None
    center_board1b = None
    img1b = None
    k_conv = None  # см (или другие единицы pc/corner_coordinates) на один шаг энкодера

    # Шаг 2: снимок Cam2
    pos2_raw = None
    center_board2 = None
    img2 = None

    running = True

    while running:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        cur_pos = get_current_position()
        cur_pos_display = "Нет данных" if cur_pos is None else str(cur_pos)

        display1 = frame1.copy()
        display2 = frame2.copy()
        draw_overlay(display1, step, cur_pos_display)
        draw_overlay(display2, step, cur_pos_display)

        cv2.imshow("Camera 1", display1)
        cv2.imshow("Camera 2", display2)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            running = False
            break

        elif key == 32:  # Пробел
            if step == 0:
                # --- Шаг 0: первый снимок Cam1 (точка A) ---
                center_board1a, rvec1a, tvec1a, corners1a = find_board_pose(frame1, mtx1, dist1)
                if center_board1a is None:
                    print("[ERROR] Доска не найдена на камере 1 (снимок A).")
                    continue
                pos1a_raw = get_current_position()

                img1a = frame1.copy()
                cv2.drawChessboardCorners(img1a, cheker_params.CHECKERBOARD, corners1a, True)
                cv2.imshow("Shot A (Cam1)", img1a)
                cv2.waitKey(500)
                cv2.destroyWindow("Shot A (Cam1)")

                pos1a_str = f"{pos1a_raw:.2f}" if pos1a_raw is not None else "Нет данных"
                print(f"[Cam1][A] Снимок сохранён. Энкодер = {pos1a_str} шагов")
                print(f"[Cam1][A] center_board (система камеры) = {center_board1a}")
                print("Теперь сдвиньте конвейер на небольшое расстояние и нажмите ПРОБЕЛ "
                      "для второго снимка Cam1 (калибровка коэффициента).")
                step = 1

            elif step == 1:
                # --- Шаг 1: второй снимок Cam1 после сдвига (точка B), расчёт k_conv ---
                if pos1a_raw is None:
                    print("[ERROR] Нет данных энкодера для точки A, коэффициент посчитать нельзя.")
                    continue

                center_board1b, rvec1b, tvec1b, corners1b = find_board_pose(frame1, mtx1, dist1)
                if center_board1b is None:
                    print("[ERROR] Доска не найдена на камере 1 (снимок B).")
                    continue
                pos1b_raw = get_current_position()
                if pos1b_raw is None:
                    print("[ERROR] Нет данных энкодера для точки B, повторите снимок.")
                    continue

                encoder_delta = pos1b_raw - pos1a_raw
                if abs(encoder_delta) < MIN_CALIB_ENCODER_DELTA:
                    print(f"[ERROR] Сдвиг энкодера слишком мал ({encoder_delta} шагов, "
                          f"нужно >= {MIN_CALIB_ENCODER_DELTA}). Сдвиньте конвейер сильнее и повторите.")
                    continue

                _, xA = project_to_conveyor(center_board1a, e1, pc1)
                _, xB = project_to_conveyor(center_board1b, e1, pc1)
                cam_delta = xB - xA

                k_conv = cam_delta / encoder_delta

                img1b = frame1.copy()
                cv2.drawChessboardCorners(img1b, cheker_params.CHECKERBOARD, corners1b, True)
                cv2.imshow("Shot B (Cam1)", img1b)
                cv2.waitKey(500)
                cv2.destroyWindow("Shot B (Cam1)")

                print(f"[Cam1][B] Снимок сохранён. Энкодер = {pos1b_raw:.2f} шагов")
                print(f"[Cam1][B] center_board (система камеры) = {center_board1b}")
                print(f"[CALIB] Смещение по камере (ex): {cam_delta}, смещение энкодера: {encoder_delta}")
                print(f"[CALIB] Коэффициент k_conv = {k_conv} (единиц расстояния / шаг энкодера)")
                print("Теперь наведите на доску камеру 2 и нажмите ПРОБЕЛ для финального снимка.")
                step = 2

            elif step == 2:
                # --- Шаг 2: снимок Cam2, финальный расчёт расстояния ---
                center_board2, rvec2, tvec2, corners2 = find_board_pose(frame2, mtx2, dist2)
                if center_board2 is None:
                    print("[ERROR] Доска не найдена на камере 2.")
                    continue
                pos2_raw = get_current_position()

                img2 = frame2.copy()
                cv2.drawChessboardCorners(img2, cheker_params.CHECKERBOARD, corners2, True)
                cv2.imshow("Shot Cam2", img2)
                cv2.waitKey(1500)
                cv2.destroyWindow("Shot Cam2")

                pos2_str = f"{pos2_raw:.2f}" if pos2_raw is not None else "Нет данных"
                print(f"[Cam2] Снимок сохранён. Энкодер = {pos2_str} шагов")
                print(f"[Cam2] center_board (система камеры) = {center_board2}")

                if (pos1a_raw is not None and pos2_raw is not None
                        and center_board1a is not None and center_board2 is not None
                        and k_conv is not None):
                    _, x_pos1 = project_to_conveyor(center_board1a, e1, pc1)
                    _, x_pos2 = project_to_conveyor(center_board2, e2, pc2)
                    print(f"x_pos1={x_pos1}, x_pos2={x_pos2}")
                    d = x_pos1 - x_pos2 + (pos2_raw - pos1a_raw) * k_conv
                    print(f"Расстояние между камерами: {d}")
                else:
                    print("[ERROR] Недостаточно данных для вычисления расстояния "
                          "(нет коэффициента k_conv или данных энкодера/доски).")

                if img1a is not None:
                    cv2.imwrite("cam1_shotA.png", img1a)
                if img1b is not None:
                    cv2.imwrite("cam1_shotB.png", img1b)
                if img2 is not None:
                    cv2.imwrite("cam2_shot.png", img2)
                print("[SAVE] Фото сохранены как cam1_shotA.png, cam1_shotB.png, cam2_shot.png")
                print("Нажмите 'r' для сброса и нового измерения, или ESC для выхода.")
                step = 3

            else:
                print("[INFO] Измерение уже завершено. Нажмите 'r' для сброса.")

        elif key == ord('r'):
            if step == 3:
                step = 0
                pos1a_raw = pos1b_raw = pos2_raw = None
                center_board1a = center_board1b = center_board2 = None
                img1a = img1b = img2 = None
                k_conv = None
                print("[RESET] Состояние сброшено.")
            else:
                print("[INFO] Сброс доступен только после завершения измерения (шаг 3).")

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
