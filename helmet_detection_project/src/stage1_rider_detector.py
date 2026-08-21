"""
stage1_rider_detector.py

Stage 1 of the pipeline: find motorcycles and riders in a frame.

Why YOLOv8n, and why zero training is needed here:
COCO (the dataset YOLOv8's released weights are trained on) already contains
a "motorcycle" class (id 3) and a "person" class (id 0). Since the very first
job of the pipeline is generic object detection -- not anything
helmet-specific -- there is nothing to gain from training a fresh model for
this stage. Using the pretrained COCO weights directly means Stage 1 works
out of the box, is fast (the 'n' = nano variant is the smallest/fastest
YOLOv8 variant, chosen for real-time performance as described in the project
brief), and is a well-understood baseline that's easy to defend in a review
("I used a pretrained detector for generic vehicle/person detection because
retraining it would just reproduce COCO's own motorcycle/person detector").

What's actually novel in this project is Stage 2 (the helmet classifier),
which cannot come pretrained because "helmet vs no-helmet on a cropped head"
is not a COCO class.
"""

from ultralytics import YOLO
from utils import person_near_motorcycle

COCO_PERSON_CLASS = 0
COCO_MOTORCYCLE_CLASS = 3


class RiderDetector:
    def __init__(self, weights_path="models/yolov8n.pt", conf_threshold=0.35):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Runs YOLOv8 on a frame and returns:
          - motorcycle_boxes: list of (x1,y1,x2,y2)
          - rider_boxes: list of person boxes that are spatially associated
            with a detected motorcycle (i.e. likely riders, not pedestrians)
        """
        results = self.model.predict(
            frame, conf=self.conf_threshold, classes=[COCO_PERSON_CLASS, COCO_MOTORCYCLE_CLASS],
            verbose=False
        )[0]

        motorcycle_boxes = []
        person_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            if cls_id == COCO_MOTORCYCLE_CLASS:
                motorcycle_boxes.append(xyxy)
            elif cls_id == COCO_PERSON_CLASS:
                person_boxes.append(xyxy)

        # Association step: a "rider" is a person box that falls in the
        # proximity zone of at least one detected motorcycle. This filters
        # out pedestrians/bystanders who happen to be in frame but are not
        # actually on a bike.
        rider_boxes = []
        for p_box in person_boxes:
            for m_box in motorcycle_boxes:
                if person_near_motorcycle(p_box, m_box):
                    rider_boxes.append(p_box)
                    break

        return motorcycle_boxes, rider_boxes
