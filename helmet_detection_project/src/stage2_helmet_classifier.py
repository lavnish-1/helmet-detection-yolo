"""
stage2_helmet_classifier.py

Stage 2 of the pipeline: given a cropped head region, classify it as
"helmet" or "no_helmet".

Why a separate CNN classifier instead of one big YOLO model trained on
"with_helmet"/"without_helmet" classes directly (the more common approach
in tutorials online):

1. Data efficiency. Head crops are small, roughly square, and centered --
   a much easier learning problem for a plain classifier than teaching a
   detector to localize a tiny helmet/head region inside a full traffic
   frame. Binary classification on a cropped, normalized input converges
   with far fewer labeled images than detection does.
2. Modularity. Stage 1 (finding riders) and Stage 2 (checking their heads)
   are logically independent problems. Keeping them separate means the
   classifier can be retrained/improved (e.g. swapped for a better backbone)
   without touching the detector, and vice versa.
3. Matches the brief. The project description explicitly separates
   "detect motorcycles/riders" from "classify whether the rider is wearing
   a helmet" as two stages -- this file implements exactly that second
   stage.

Why MobileNetV2 as the backbone (via transfer learning):
It's small enough for real-time inference on a laptop CPU/low-end GPU,
and transfer learning from ImageNet weights means the model already has
useful low-level features (edges, textures, curved shapes) so it needs only
a modest labeled dataset (a few hundred images per class is enough to get
a usable classifier) to fine-tune the final layers for this specific task.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

CLASS_NAMES = ["with_helmet", "without_helmet"]

IMG_SIZE = 128

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(pretrained=True):
    """MobileNetV2 with its classifier head replaced for binary output."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


class HelmetClassifier:
    """Inference wrapper: loads trained weights and classifies head crops."""

    def __init__(self, weights_path="models/helmet_classifier.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, head_crop_bgr):
        """
        head_crop_bgr: a numpy array (H, W, 3) in BGR order, as returned by
        OpenCV's frame slicing. Returns (label:str, confidence:float).
        """
        rgb = head_crop_bgr[:, :, ::-1]  # BGR -> RGB
        pil_img = Image.fromarray(np.ascontiguousarray(rgb))
        tensor = inference_transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        return CLASS_NAMES[pred_idx], float(probs[pred_idx])
