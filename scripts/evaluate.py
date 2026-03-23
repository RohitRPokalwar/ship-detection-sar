"""
Evaluation Script — computes precision, recall, F1, mAP.
Usage: python scripts/evaluate.py --weights models/yolov8n_sar.pt --data data/yolo_format/dataset.yaml
"""
import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(args):
    from ultralytics import YOLO
    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        verbose=True,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }
    metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / \
                    (metrics["precision"] + metrics["recall"] + 1e-6)

    print("\n" + "="*50)
    print("📊 EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")
    print("="*50)

    out = Path(args.output)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to: {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="outputs/eval_metrics.json")
    args = parser.parse_args()
    evaluate(args)
