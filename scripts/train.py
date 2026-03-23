"""
YOLOv8n Training Script for SAR Ship Detection.
Usage: python scripts/train.py --data data/yolo_format/dataset.yaml --epochs 50
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_dataset_yaml(data_dir: str, output_path: str):
    """Create YOLO dataset YAML configuration."""
    yaml_content = f"""
path: {data_dir}
train: images/train
val: images/val

nc: 1
names: ['ship']
"""
    Path(output_path).write_text(yaml_content.strip())
    print(f"Dataset YAML created: {output_path}")
    return output_path


def train(args):
    from ultralytics import YOLO
    from config.config import MODELS_DIR, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_IMAGE_SIZE

    model = YOLO(args.model)
    dataset_yaml = args.data
    if not Path(dataset_yaml).exists():
        dataset_yaml = create_dataset_yaml(args.data, str(Path(args.data) / "dataset.yaml"))

    results = model.train(
        data=dataset_yaml,
        epochs=args.epochs or TRAIN_EPOCHS,
        batch=args.batch or TRAIN_BATCH_SIZE,
        imgsz=args.imgsz or TRAIN_IMAGE_SIZE,
        device=args.device,
        project=str(MODELS_DIR / "runs"),
        name="yolov8n_sar",
        patience=10,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights
    best_weights = MODELS_DIR / "runs" / "yolov8n_sar" / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        dest = MODELS_DIR / "yolov8n_sar.pt"
        shutil.copy2(str(best_weights), str(dest))
        print(f"\n✅ Best weights saved to: {dest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for SAR ship detection")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML or directory path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    train(args)
