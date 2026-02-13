from pathlib import Path
import math

import cv2
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================

# Видео
SOURCE_VIDEO = Path.home() / "Downloads" / "vehicle-counting.mp4"
OUTPUT_VIDEO = Path.home() / "Movies" / "vehicle-counting-output.mp4"

# Модель
MODEL = "yolov8s.pt"

# Линия подсчёта
LINE_START = (475, 1822)
LINE_END   = (1731, 1822)

# Полоса вокруг линии (px)
LINE_BAND_PX = 35

# Классы транспорта (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# Детекция/трекинг параметры
CONF = 0.18
IOU = 0.55

# Улучшение стабильности
IMGSZ = 1280

# Толщина/видимость
LINE_THICKNESS = 10
BOX_THICKNESS = 6
TEXT_THICKNESS = 6
ID_FONT_SCALE = 0.9
COUNT_FONT_SCALE = 1.5

# Считаем ТОЛЬКО встречку
COUNT_DIRECTION = "down"
DIR_MIN_PX = 2  # минимальный сдвиг, чтобы шум не считался

# Трекер: BoT-SORT часто стабильнее на плотных сценах, чем ByteTrack
TRACKER = "botsort.yaml"

# =========================
# HELPERS
# =========================

def side_of_line(p, a, b) -> int:
    """С какой стороны от линии a-b находится точка p: -1 / 0 / 1"""
    x, y = p
    x1, y1 = a
    x2, y2 = b
    v = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def point_line_distance_px(p, a, b) -> float:
    """Расстояние от точки p до прямой через a-b (в пикселях)."""
    x, y = p
    x1, y1 = a
    x2, y2 = b
    num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = math.hypot((y2 - y1), (x2 - x1))
    return num / (den + 1e-9)


def label_position_near_line(a, b, offset_px=35):
    """Позиция текста около середины линии, чуть выше."""
    mx = (a[0] + b[0]) // 2
    my = (a[1] + b[1]) // 2
    return (mx, max(0, my - offset_px))


def direction_ok(dx, dy, mode: str, min_px: int) -> bool:
    if mode == "down":
        return dy > min_px
    if mode == "up":
        return dy < -min_px
    if mode == "right":
        return dx > min_px
    if mode == "left":
        return dx < -min_px
    return True

# =========================
# MAIN
# =========================

def main():
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не могу открыть видео: {SOURCE_VIDEO}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    model = YOLO(MODEL)

    counted_ids = set()   # чтобы не считать один и тот же track_id дважды
    prev_side = {}        # tid -> -1/0/1 (для определения пересечения)
    prev_point = {}       # tid -> (x, y) (для направления)

    total_count = 0
    label_pos = label_position_near_line(LINE_START, LINE_END, offset_px=35)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # TRACKING: даёт боксы + устойчивые ID
        if TRACKER:
            results = model.track(
                frame, persist=True, conf=CONF, iou=IOU, imgsz=IMGSZ,
                tracker=TRACKER, verbose=False
            )
        else:
            results = model.track(
                frame, persist=True, conf=CONF, iou=IOU, imgsz=IMGSZ,
                verbose=False
            )

        r = results[0]

        # линия подсчёта
        cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), LINE_THICKNESS)

        # если есть боксы и ID
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, tid in zip(boxes, cls, ids):
                if c not in VEHICLE_CLASSES:
                    continue

                # --- 1) ВИЗУАЛИЗАЦИЯ: рисуем ВСЕ машины ---
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 255, 0), BOX_THICKNESS
                )
                cv2.putText(
                    frame, f"ID {tid}",
                    (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, ID_FONT_SCALE,
                    (0, 255, 0), TEXT_THICKNESS
                )

                # --- 2) ТОЧКА ДЛЯ ПОДСЧЁТА: низ бокса ---
                px = int((x1 + x2) / 2)
                py = int(y2)
                p = (px, py)

                # направление движения (dx/dy)
                dx = dy = 0
                if tid in prev_point:
                    dx = p[0] - prev_point[tid][0]
                    dy = p[1] - prev_point[tid][1]
                prev_point[tid] = p

                # геометрия пересечения/касания линии
                cur_side = side_of_line(p, LINE_START, LINE_END)
                dist = point_line_distance_px(p, LINE_START, LINE_END)

                if tid not in counted_ids and tid in prev_side:
                    ps = prev_side[tid]

                    crossed = (ps != 0 and cur_side != 0 and ps != cur_side and dist <= LINE_BAND_PX)
                    correct_dir = direction_ok(dx, dy, COUNT_DIRECTION, DIR_MIN_PX)

                    if crossed and correct_dir:
                        total_count += 1
                        counted_ids.add(tid)

                prev_side[tid] = cur_side

        # счётчик возле линии
        cv2.putText(
            frame, f"Total (oncoming): {total_count}",
            label_pos, cv2.FONT_HERSHEY_SIMPLEX,
            COUNT_FONT_SCALE, (255, 255, 255), TEXT_THICKNESS
        )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"TOTAL ONCOMING: {total_count}")
    print(f"Готово! Результат: {OUTPUT_VIDEO.resolve()}")


if __name__ == "__main__":
    main()