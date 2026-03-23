"""
Convert SSDD VOC-format annotations to YOLO format and set up dataset directory.
Usage: python scripts/convert_annotations.py
"""
import xml.etree.ElementTree as ET
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SSDD_ROOT = Path(__file__).resolve().parent.parent / "data" / "Official-SSDD-OPEN" / "BBox_SSDD" / "voc_style"
YOLO_ROOT = Path(__file__).resolve().parent.parent / "data" / "yolo_format"

CLASS_MAP = {"ship": 0}


def voc_to_yolo(xml_path: Path, img_width: int, img_height: int):
    """Convert a single VOC XML annotation to YOLO format lines."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    lines = []

    # Try to get actual image size from XML
    size = root.find("size")
    if size is not None:
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        if w > 0 and h > 0:
            img_width, img_height = w, h

    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        cls_id = CLASS_MAP.get(name, 0)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert to YOLO format: class x_center y_center width height (all normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # Clamp values
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def convert_split(split: str):
    """Convert a train/test split from VOC to YOLO format."""
    img_src = SSDD_ROOT / f"JPEGImages_{split}"
    ann_src = SSDD_ROOT / f"Annotations_{split}"

    img_dst = YOLO_ROOT / "images" / split
    lbl_dst = YOLO_ROOT / "labels" / split

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    converted = 0
    skipped = 0

    for img_path in images:
        stem = img_path.stem
        xml_path = ann_src / f"{stem}.xml"

        # Copy image
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            shutil.copy2(str(img_path), str(dst_img))

        # Convert annotation
        if xml_path.exists():
            lines = voc_to_yolo(xml_path, 640, 640)  # default fallback; XML has actual size
            lbl_path = lbl_dst / f"{stem}.txt"
            lbl_path.write_text("\n".join(lines))
            converted += 1
        else:
            # No annotation → empty label file (background image)
            lbl_path = lbl_dst / f"{stem}.txt"
            lbl_path.write_text("")
            skipped += 1

    print(f"  [{split}] Converted: {converted}, No annotation: {skipped}, Total images: {len(images)}")
    return converted


def create_yaml():
    """Create YOLO dataset YAML file."""
    yaml_path = YOLO_ROOT / "dataset.yaml"
    yaml_content = f"""path: {YOLO_ROOT.as_posix()}
train: images/train
val: images/test

nc: 1
names: ['ship']
"""
    yaml_path.write_text(yaml_content)
    print(f"  Dataset YAML: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    print("=" * 60)
    print("SSDD VOC → YOLO Format Converter")
    print("=" * 60)
    print(f"Source: {SSDD_ROOT}")
    print(f"Output: {YOLO_ROOT}")
    print()

    YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    print("Converting train split...")
    convert_split("train")
    print("Converting test split...")
    convert_split("test")
    print()

    yaml_path = create_yaml()
    print()
    print("✅ Conversion complete!")
    print(f"   Train: {YOLO_ROOT / 'images' / 'train'}")
    print(f"   Val:   {YOLO_ROOT / 'images' / 'test'}")
    print(f"   YAML:  {yaml_path}")
    print()
    print("To train: python scripts/train.py --data data/yolo_format/dataset.yaml")
