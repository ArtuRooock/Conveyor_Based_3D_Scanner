import cv2 as cv
import numpy as np
import open3d as o3d


def get_click_coordinates(img: cv.Mat, label: str):
    coords = {"x": None, "y": None}

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            coords["x"] = x
            coords["y"] = y

    cv.imshow(label, img)
    cv.setMouseCallback(label, mouse_callback)

    while coords["x"] is None:
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()

    if coords["x"] is None:
        return None

    return coords["x"], coords["y"]


def cutter(frame: cv.Mat,
           point1: tuple[int, int],
           point2: tuple[int, int],
           ) -> cv.Mat:
    x1, x2 = point1[0], point2[0]
    y = max(point1[1], point2[1])
    x1, x2 = sorted((x1, x2))
    x1 = max(0, x1)
    x2 = min(frame.shape[0], x2)
    return frame[0:y, x1:x2]


def compute_row_average_x_fast(gray: np.ndarray) -> np.ndarray:
    mask = gray > 0
    height, width = gray.shape
    x_indices = np.arange(width)

    sum_x = (mask * x_indices).sum(axis=1)
    count = mask.sum(axis=1)

    valid = count > 0
    avg_x = np.zeros(height, dtype=np.int32)
    avg_x[valid] = (sum_x[valid] / count[valid]).astype(np.int32)

    y_indices = np.arange(height)
    points = np.stack([avg_x[valid], y_indices[valid]], axis=1)

    return points


def laser_detector(gr: cv.Mat, lowerQ: int, upperQ: int):
    gr = cv.bitwise_and(
        gr, gr, mask=cv.inRange(gr, lowerQ, upperQ))
    points = compute_row_average_x_fast(gr)
    cv.imshow("INPUT", gr)
    lasered = cv.Mat(np.ndarray(gr.shape, dtype=np.uint8))
    for point in points:
        cv.circle(lasered, point, 1, (255, 0, 0), 1)
    cv.imshow("LASERD", lasered)
    return points


def add_z_dimension(points_2d: np.ndarray, z: float) -> np.ndarray:
    """Расширяет массив (N, 2) до (N, 3), добавляя постоянную координату z."""
    z_column = np.full((points_2d.shape[0], 1), z, dtype=np.float64)
    return np.hstack([points_2d, z_column])


def save_to_pts(points_xyz: np.ndarray, filename: str):
    """
    Сохраняет облако точек в формат .pts (текстовый, X Y Z).
    Ожидается массив формы (N, 3).
    """
    if points_xyz.shape[1] != 3:
        raise ValueError("Массив должен иметь размерность (N, 3)")
    np.savetxt(filename, points_xyz, fmt='%.6f', delimiter=' ',
               header='', comments='', encoding='utf-8')


if __name__ == "__main__":
    # --- ИСПОЛЬЗУЕМ КАМЕРУ ---
    cap = cv.VideoCapture(1)          # 0 – первая веб-камера, 1 – вторая и т.д.
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Попытка установить желаемые параметры (не критично, если не поддерживаются)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
    cap.set(cv.CAP_PROP_FPS, 120)

    # Создаём визуализатор Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # заглушка
    vis.add_geometry(pcd)

    # Получаем первый кадр для выбора области обрезки
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from camera")
    upper_point = get_click_coordinates(frame, "Upper cut")
    lower_point = get_click_coordinates(frame, "Lower cut")
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cutter(frame, lower_point, upper_point)

    z = 0
    delta = 6
    frame_count = 0
    all_points = []  # храним точки в формате (z, y, x)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera stream ended")
            break

        cv.imshow("row", frame)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_cut = cutter(frame_gray, lower_point, upper_point)

        try:
            points = laser_detector(frame_cut, 250, 255)
            if points.size > 0:
                result = add_z_dimension(points, z)
                # Переставляем столбцы: (z, y, x) – так удобнее для визуализации в Open3D
                points_xzy = result[:, [2, 1, 0]]
                all_points.append(points_xzy)

                # Обновляем облако точек
                pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                # vis.reset_view_point(True)  # раскомментируйте, если хотите сбрасывать вид

                z += delta
                frame_count += 1
        except Exception as e:
            print(f"Processing error: {e}, skipping frame")
            continue

        # Обработка клавиш
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC – выход
            break
        elif key == ord('s'):  # S – сохранить снимок в .pts
            if all_points:
                merged = np.vstack(all_points)           # (N, 3) в формате (z,y,x)
                points_xyz = merged[:, [2, 1, 0]]        # переставляем в (x,y,z)
                save_to_pts(points_xyz, "snapshot.pts")
                print(f"Snapshot saved: {len(points_xyz)} points")
            else:
                print("No points to save yet")

    # --- Завершение работы ---
    # Сохраняем итоговое облако, если есть точки
    if all_points:
        merged = np.vstack(all_points)
        points_xyz = merged[:, [2, 1, 0]]
        save_to_pts(points_xyz, "output2.pts")
        print(f"Final cloud saved: {len(points_xyz)} points")
    else:
        print("No points collected")

    cv.destroyAllWindows()
    vis.destroy_window()
    cap.release()