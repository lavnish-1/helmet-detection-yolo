"""
utils.py
Shared helper functions used across the pipeline:
- box_iou / box_overlap_ratio: measure how much two bounding boxes overlap
- get_head_crop: crop the head region out of a person's bounding box
- draw_label: consistent, readable annotation drawing
- ViolationLogger: writes every no-helmet detection to a CSV with a timestamp
  and saves a snapshot image, so the project produces an auditable record
  (this is what turns "detects it on screen" into "usable for surveillance").
"""

import os
import csv
from datetime import datetime

import cv2


def box_iou(boxA, boxB):
    """Standard Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def person_near_motorcycle(person_box, moto_box, x_margin_ratio=0.6):
    """
    A rider's body rarely overlaps the motorcycle box much (IoU is often near
    zero because the bike is below/behind the person). What actually signals
    "this person is riding this bike" is horizontal proximity plus vertical
    stacking (person's feet are around the bike's seat height). So instead of
    relying on IoU alone, we widen the motorcycle box horizontally and check
    if the person's box center falls inside that widened zone.
    """
    mx1, my1, mx2, my2 = moto_box
    moto_w = mx2 - mx1
    margin = moto_w * x_margin_ratio
    zone = (mx1 - margin, my1 - moto_w, mx2 + margin, my2 + (my2 - my1) * 3)

    px_center = (person_box[0] + person_box[2]) / 2
    py_center = (person_box[1] + person_box[3]) / 2

    return zone[0] <= px_center <= zone[2] and zone[1] <= py_center <= zone[3]


def get_head_crop(frame, person_box, head_fraction=0.35):
    """
    Crop the top `head_fraction` of a person's bounding box. This is a cheap
    but effective heuristic: heads are consistently near the top of a person
    detection box regardless of pose, and it avoids needing a separate
    keypoint/pose model just to locate the head.
    """
    x1, y1, x2, y2 = [int(v) for v in person_box]
    h = y2 - y1
    head_y2 = y1 + int(h * head_fraction)
    x1, y1 = max(0, x1), max(0, y1)
    head_y2 = min(frame.shape[0], head_y2)
    x2 = min(frame.shape[1], x2)
    if x2 <= x1 or head_y2 <= y1:
        return None
    return frame[y1:head_y2, x1:x2]


def draw_label(frame, box, text, color, thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)


class ViolationLogger:
    """Logs every 'no helmet' event once per tracked violation, with a
    timestamp and a saved snapshot, into outputs/violations.csv."""

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        self.snap_dir = os.path.join(output_dir, "violation_snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, "violations.csv")
        is_new = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.csv_file)
        if is_new:
            self.writer.writerow(["timestamp", "frame_number", "confidence", "snapshot_path"])
        self._count = 0

    def log(self, frame, frame_number, confidence, crop_box):
        self._count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snap_name = f"violation_{self._count:05d}.jpg"
        snap_path = os.path.join(self.snap_dir, snap_name)
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        cv2.imwrite(snap_path, frame[max(0, y1):y2, max(0, x1):x2])
        self.writer.writerow([timestamp, frame_number, f"{confidence:.3f}", snap_path])
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
