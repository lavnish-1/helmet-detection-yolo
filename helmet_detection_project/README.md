# Helmet Detection in Motorcycles using Machine Learning

Real-time video pipeline that detects motorcycles, identifies riders, and
classifies whether each rider is wearing a helmet — triggering and logging
an alert when one isn't. Built to match the two-stage architecture in the
project brief: **detection** (motorcycle/rider) → **classification**
(helmet/no-helmet).

## Architecture & why it's built this way

```
video frame
   |
   v
[Stage 1: YOLOv8n, COCO-pretrained]  --------> motorcycle boxes + person boxes
   |                                                   |
   |  (proximity heuristic: is this person             |
   |   near a detected motorcycle?)                    |
   v                                                    v
rider boxes  <-----------------------------------------+
   |
   v
[head crop: top 35% of rider box]
   |
   v
[Stage 2: MobileNetV2, fine-tuned]  --------> "with_helmet" / "without_helmet" + confidence
   |
   v
annotate frame, and if "without_helmet": log timestamp + save snapshot + alert
```

**Stage 1 — motorcycle/rider detection (`stage1_rider_detector.py`)**
Uses YOLOv8n with its official COCO-pretrained weights, unmodified. COCO
already includes `person` and `motorcycle` as classes, so this stage needs
**zero training** — there's nothing helmet-specific about "find the bikes
and the people," so retraining a detector for it would just re-derive what
COCO's own weights already do. The "nano" (`n`) variant is the smallest/
fastest YOLOv8 size, chosen because the brief calls for real-time
performance. A person box only becomes a "rider" box if it falls in a
proximity zone around a detected motorcycle (see `person_near_motorcycle`
in `utils.py`) — this filters out pedestrians who are simply in frame.

**Stage 2 — helmet classification (`stage2_helmet_classifier.py`,
`train_helmet_classifier.py`)**
This is the actual novel/trained part of the project. It's a binary image
classifier (MobileNetV2, ImageNet-pretrained, fine-tuned) run on the
*cropped head region* of each rider, not a second object detector. Three
reasons for that design choice, worth having ready if asked:
1. **Data efficiency** — classifying a small, centered, cropped image is a
   much easier learning problem than localizing a tiny helmet region
   inside a full traffic scene, so it needs far fewer labeled examples to
   reach usable accuracy.
2. **Modularity** — detection and classification are independent problems.
   The classifier can be retrained or swapped for a different backbone
   without touching the detector.
3. **Directly matches the brief** — "detect motorcycles/riders" and "then
   classify whether the rider has a helmet" are described as two separate
   stages, and this is a literal implementation of that.

Head localization uses a simple, defensible heuristic (top 35% of the
person's bounding box) rather than a pose/keypoint model — heads are
reliably near the top of a person detection regardless of pose, so this
avoids the cost and complexity of a third model for a problem a heuristic
already solves well.

## Project structure

```
helmet_detection_project/
├── README.md
├── requirements.txt
├── models/
│   ├── yolov8n.pt              <- included, pretrained, ready to use
│   └── (helmet_classifier.pt   <- created by you, see Setup step 3)
├── data/
│   ├── with_helmet/            <- put training images here
│   └── without_helmet/         <- put training images here
├── src/
│   ├── utils.py                 shared helpers (IoU, head-crop, logging)
│   ├── stage1_rider_detector.py Stage 1: YOLOv8 motorcycle/rider detection
│   ├── stage2_helmet_classifier.py  Stage 2: classifier model + inference
│   ├── train_helmet_classifier.py   trains Stage 2 on your dataset
│   └── pipeline.py               ties both stages together, runs on video
└── outputs/                     annotated videos + violations.csv land here
```

## Setup (VS Code)

1. **Open the folder in VS Code**, then create a virtual environment and
   install dependencies:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
   `models/yolov8n.pt` is already included, so Stage 1 works immediately —
   no download or training needed for it.

2. **Get a labeled dataset for the helmet classifier.** This is the one
   piece that can't come pretrained, because "helmet vs no-helmet on a
   cropped head" isn't a standard COCO class. Options, easiest first:
   - Download the **Kaggle "Helmet Detection" dataset**
     (`andrewmvd/helmet-detection`) — it has bounding-box annotations;
     crop each box out to build `with_helmet/` and `without_helmet/`
     folders of plain images.
   - Or a Roboflow Universe helmet/PPE dataset exported as classification
     (not detection) format, which drops straight into the two folders.
   - Or crop your own head images from footage using `get_head_crop()` in
     `utils.py` and manually sort them.

   Aim for **150–200+ images per class** to start. Put them in
   `data/with_helmet/` and `data/without_helmet/`.

3. **Train the classifier:**
   ```bash
   cd src
   python train_helmet_classifier.py --epochs 15
   ```
   This saves `models/helmet_classifier.pt`. Watch `val_acc` in the
   printed log — if it's low, the usual first fixes are more data, more
   varied lighting/angles in the images, or a few more epochs.

4. **Run detection on a video:**
   ```bash
   python pipeline.py --source path/to/video.mp4
   # or use a webcam:
   python pipeline.py --source 0
   ```
   Outputs land in `../outputs/`:
   - `annotated_output.mp4` — the video with green ("Helmet") / red
     ("NO HELMET") boxes drawn per rider, frame by frame.
   - `violations.csv` — timestamp, frame number, confidence, and a
     snapshot path for every no-helmet detection (the "alert" the brief
     describes, made auditable rather than just a screen flash).
   - `violation_snapshots/` — the cropped image saved for each violation.

   Useful flags: `--det_conf` (YOLO detection confidence, default 0.35),
   `--cls_conf` (helmet-classifier confidence, default 0.55 — below this
   it's treated as a violation, since an uncertain classification on a
   safety check should err toward flagging it), `--skip_frames N` to speed
   up processing on long videos by only running inference on every
   (N+1)th frame.

## Verified working

Every module (`stage1_rider_detector.py`, `stage2_helmet_classifier.py`,
`train_helmet_classifier.py`, `pipeline.py`, `utils.py`) was executed
end-to-end during development: Stage 1 loads and runs YOLOv8 detection,
the training loop runs a full epoch cycle on a sample dataset and saves
weights, and `pipeline.py` runs a video through both stages and produces
`annotated_output.mp4` + `violations.csv`. The only step left for you to
do locally is point `train_helmet_classifier.py` at a *real* labeled
dataset (step 2 above) so the classifier learns actual helmet/no-helmet
patterns instead of placeholder data.

## Possible extensions (good "future work" talking points)

- Add a license-plate detector (e.g. another YOLO stage) on flagged
  violation frames, to make the alert actionable for enforcement.
- Multi-object tracking (e.g. ByteTrack, built into `ultralytics`) so a
  rider is only logged once per violation instead of once per frame.
- Export the classifier to ONNX/TensorRT for faster edge deployment.
- Add a second class for partial helmet use (e.g. unstrapped) if your
  dataset supports it.
