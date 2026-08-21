"""
prepare_dataset.py

Converts a Pascal-VOC-annotated helmet dataset (e.g. the Kaggle
"Helmet Detection" dataset, or a Roboflow Universe export in VOC format)
into the plain cropped-image folders that train_helmet_classifier.py
expects:
    data/with_helmet/*.jpg
    data/without_helmet/*.jpg

This script does NOT assume a fixed folder layout. It recursively searches
--archive_dir for every .xml file, wherever it is nested (this covers both
common shapes you'll run into):

  Kaggle style:                      Roboflow style:
    archive/                           my_dataset/
      images/*.png                       train/*.jpg + train/*.xml
      annotations/*.xml                  valid/*.jpg + valid/*.xml
                                          test/*.jpg  + test/*.xml

For each XML, it looks for the matching image (a) by the <filename> tag
inside the XML, checked first in the same folder as the XML, then (b) by
matching the XML's own filename stem against common image extensions, also
checked in the same folder first, then anywhere else under --archive_dir as
a last resort.

Each <object>'s <name> tag (e.g. "With Helmet" / "Without Helmet", any
casing, with dashes/underscores/spaces) is classified as with_helmet /
without_helmet / skipped (anything else, e.g. "motorbike", "licence").
The corresponding <bndbox> is cropped out of the image and saved.

Usage (run from the src/ folder, after extracting the dataset zip):
    python prepare_dataset.py --archive_dir path/to/extracted_folder --out_dir ../data
"""

import argparse
import os
import xml.etree.ElementTree as ET

from PIL import Image

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def classify_label(name: str):
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    if "without" in n or "no_helmet" in n or n == "no_helmet":
        return "without_helmet"
    if "with" in n and "without" not in n:
        return "with_helmet"
    return None  # e.g. "motorbike", "licence", "rider" -- not a helmet label


def find_all_xml(root_dir):
    xml_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".xml"):
                xml_paths.append(os.path.join(dirpath, f))
    return xml_paths


def find_image_for_xml(xml_path, archive_dir, filename_hint):
    xml_dir = os.path.dirname(xml_path)
    stem = os.path.splitext(os.path.basename(xml_path))[0]

    candidates = []
    if filename_hint:
        candidates.append(os.path.join(xml_dir, filename_hint))
    for ext in IMG_EXTS:
        candidates.append(os.path.join(xml_dir, stem + ext))

    for c in candidates:
        if os.path.exists(c):
            return c

    # last resort: search the whole archive for a file with this stem
    for dirpath, _, filenames in os.walk(archive_dir):
        for f in filenames:
            if os.path.splitext(f)[0] == stem and f.lower().endswith(IMG_EXTS):
                return os.path.join(dirpath, f)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir", required=True,
                         help="Folder containing the extracted dataset (any layout - "
                              "the script searches it recursively for .xml files)")
    parser.add_argument("--out_dir", default="../data")
    parser.add_argument("--pad", type=int, default=6,
                         help="Pixels of padding added around each box before cropping")
    args = parser.parse_args()

    if not os.path.isdir(args.archive_dir):
        raise FileNotFoundError(f"'{args.archive_dir}' is not a folder that exists.")

    with_dir = os.path.join(args.out_dir, "with_helmet")
    without_dir = os.path.join(args.out_dir, "without_helmet")
    os.makedirs(with_dir, exist_ok=True)
    os.makedirs(without_dir, exist_ok=True)

    xml_files = find_all_xml(args.archive_dir)
    print(f"Found {len(xml_files)} annotation (.xml) files under {args.archive_dir}")

    if len(xml_files) == 0:
        print("No .xml files found anywhere in that folder. Double-check --archive_dir "
              "points at the extracted dataset (not the .zip file itself), and that you "
              "downloaded a Pascal VOC ('voc') export, not YOLO/COCO.")
        return

    counts = {"with_helmet": 0, "without_helmet": 0, "skipped": 0}
    missing_images = 0

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            print(f"  [skip] could not parse {xml_path}")
            continue
        root = tree.getroot()

        filename_tag = root.find("filename")
        filename_hint = filename_tag.text if filename_tag is not None else None

        image_path = find_image_for_xml(xml_path, args.archive_dir, filename_hint)
        if image_path is None:
            missing_images += 1
            continue

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        stem = os.path.splitext(os.path.basename(xml_path))[0]

        for i, obj in enumerate(root.findall("object")):
            name_tag = obj.find("name")
            if name_tag is None or not name_tag.text:
                continue
            label = classify_label(name_tag.text)
            if label is None:
                counts["skipped"] += 1
                continue

            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = int(float(box.find("xmin").text)) - args.pad
            ymin = int(float(box.find("ymin").text)) - args.pad
            xmax = int(float(box.find("xmax").text)) + args.pad
            ymax = int(float(box.find("ymax").text)) + args.pad
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img.crop((xmin, ymin, xmax, ymax))
            out_folder = with_dir if label == "with_helmet" else without_dir
            out_path = os.path.join(out_folder, f"{stem}_{i}.jpg")
            crop.save(out_path, quality=95)
            counts[label] += 1

    print(f"Done. with_helmet={counts['with_helmet']} images, "
          f"without_helmet={counts['without_helmet']} images, "
          f"skipped_other_classes={counts['skipped']}, "
          f"xml_with_no_matching_image={missing_images}")
    print(f"Saved into {with_dir} and {without_dir}")


if __name__ == "__main__":
    main()
