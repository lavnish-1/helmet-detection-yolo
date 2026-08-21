"""
pipeline.py

End-to-end video pipeline described in the project brief:
  1. Read input video frame by frame.
  2. Stage 1: detect motorcycles, and persons who are riding them.
  3. For each rider, crop the head region.
  4. Stage 2: classify the head crop as helmet / no_helmet.
  5. If no_helmet: draw a red box + alert, and log it (timestamp + snapshot).
  6. Write an annotated output video so results can be reviewed frame by frame.

Usage:
    python pipeline.py --source path/to/video.mp4
    python pipeline.py --source 0            # webcam
"""

import argparse
import os
import time

import cv2

from stage1_rider_detector import RiderDetector
from stage2_helmet_classifier import HelmetClassifier
from utils import get_head_crop, draw_label, ViolationLogger

GREEN = (0, 200, 0)
RED = (0, 0, 220)
YELLOW = (0, 200, 200)


def run(source, weights_dir="../models", output_path="../outputs/annotated_output.mp4",
        conf_threshold=0.35, classifier_conf_threshold=0.55, skip_frames=0):

    yolo_weights = os.path.join(weights_dir, "yolov8n.pt")
    classifier_weights = os.path.join(weights_dir, "helmet_classifier.pt")

    if not os.path.exists(classifier_weights):
        raise FileNotFoundError(
            f"No trained classifier found at {classifier_weights}.\n"
            f"Run train_helmet_classifier.py first (see its docstring for how "
            f"to get a labeled dataset)."
        )

    detector = RiderDetector(weights_path=yolo_weights, conf_threshold=conf_threshold)
    classifier = HelmetClassifier(weights_path=classifier_weights)
    logger = ViolationLogger(output_dir=os.path.dirname(output_path) or "outputs")

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_number = 0
    violations_this_run = 0
    start_time = time.time()

    print(f"Processing {source} -> {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        if skip_frames and frame_number % (skip_frames + 1) != 0:
            writer.write(frame)
            continue

        motorcycle_boxes, rider_boxes = detector.detect(frame)

        for m_box in motorcycle_boxes:
            draw_label(frame, m_box, "motorcycle", YELLOW, thickness=1)

        for r_box in rider_boxes:
            head_crop = get_head_crop(frame, r_box)
            if head_crop is None or head_crop.size == 0:
                continue

            label, confidence = classifier.predict(head_crop)

            if label == "with_helmet" and confidence >= classifier_conf_threshold:
                draw_label(frame, r_box, f"Helmet ({confidence:.2f})", GREEN)
            else:
                draw_label(frame, r_box, f"NO HELMET ({confidence:.2f})", RED)
                logger.log(frame, frame_number, confidence, r_box)
                violations_this_run += 1

        writer.write(frame)

        if frame_number % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  frame {frame_number} | {frame_number/elapsed:.1f} fps | "
                  f"{violations_this_run} violations so far")

    cap.release()
    writer.release()
    logger.close()

    print(f"Done. {frame_number} frames processed, {violations_this_run} violations logged.")
    print(f"Annotated video: {output_path}")
    print(f"Violation log:   {os.path.join(os.path.dirname(output_path) or 'outputs', 'violations.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet detection pipeline")
    parser.add_argument("--source", required=True, help="Path to video file, or 0 for webcam")
    parser.add_argument("--weights_dir", default="../models")
    parser.add_argument("--output", default="../outputs/annotated_output.mp4")
    parser.add_argument("--det_conf", type=float, default=0.35, help="YOLO detection confidence threshold")
    parser.add_argument("--cls_conf", type=float, default=0.55, help="Helmet classifier confidence threshold")
    parser.add_argument("--skip_frames", type=int, default=0,
                         help="Process 1 of every (skip_frames+1) frames, for speed on long videos")
    args = parser.parse_args()

    run(args.source, args.weights_dir, args.output, args.det_conf, args.cls_conf, args.skip_frames)
