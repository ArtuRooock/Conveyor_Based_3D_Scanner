import cv2
from pathlib import Path


INPUT_VIDEO_PATH = r"C:\Users\Artem\Yandex.Disk\МАИ\2 курс\4 семестр\Практика\3D_Scanner\video4scaner\step4_scan.avi"
OUTPUT_DIR = r"C:\Users\Artem\Yandex.Disk\МАИ\2 курс\4 семестр\Практика\3D_Scanner\video4scaner\examples\code\img\calib4"
FRAME_INTERVAL = 20

def save_frame_unicode(frame, save_path: Path) -> bool:
    ret, buf = cv2.imencode('.jpg', frame)
    if ret:
        save_path.write_bytes(buf.tobytes())
        return True
    return False

def extract_frames(video_path: str, output_dir: str, interval: int) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Видеофайл не найден: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    frame_idx = 0
    saved_count = 0

    print(f"Начинаем обработку: {video_path.name}")
    print(f"Интервал сохранения: каждый {interval}-й кадр\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            filename = f"china_{saved_count:03d}.png"
            save_path = output_dir / filename

            success = save_frame_unicode(frame, save_path)
            if success:
                saved_count += 1
                print(f"  Сохранён: {save_path.name}  (кадр #{frame_idx})")
            else:
                print(f"  Ошибка сохранения: {save_path.name}")

        frame_idx += 1

    cap.release()
    print(f"\nГотово. Обработано кадров: {frame_idx}, сохранено: {saved_count}")
    print(f"Все кадры находятся в: {output_dir.resolve()}")

if __name__ == "__main__":
    extract_frames(INPUT_VIDEO_PATH, OUTPUT_DIR, FRAME_INTERVAL)

