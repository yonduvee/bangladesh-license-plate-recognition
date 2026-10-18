from pathlib import Path
from datetime import datetime
import shutil
import random
import sys

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "yolo" / "Character Recognition"
BASE_MODEL = PROJECT_ROOT / "models" / "character" / "character_best.pt"
CLASS_TO_BALANCE = 20
CLASS_NAME_TO_BALANCE = "ka"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def find_image(image_dir, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def read_label_lines(label_path):
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_target_class(lines):
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            try:
                if int(parts[0]) == CLASS_TO_BALANCE:
                    return True
            except ValueError:
                pass
    return False


def count_class_objects(labels_dir, class_id):
    count = 0
    for label_path in labels_dir.glob("*.txt"):
        for line in read_label_lines(label_path):
            parts = line.split()
            if len(parts) == 5 and parts[0].isdigit() and int(parts[0]) == class_id:
                count += 1
    return count


def make_photometric_variant(image, rng):
    alpha = float(rng.uniform(0.88, 1.14))
    beta = int(rng.integers(-10, 11))
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gamma = float(rng.uniform(0.90, 1.10))
    lookup = np.array([min(255, max(0, round(((value / 255.0) ** (1.0 / gamma)) * 255.0))) for value in range(256)], dtype=np.uint8)
    return cv2.LUT(adjusted, lookup)


def build_experiment_dataset(experiment_root, original_data):
    for split in ["train", "validation", "test"]:
        source_images = DATASET_ROOT / "images" / split
        source_labels = DATASET_ROOT / "labels" / split
        target_images = experiment_root / "images" / split
        target_labels = experiment_root / "labels" / split
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Missing images or labels for split: {split}")
        shutil.copytree(source_images, target_images)
        shutil.copytree(source_labels, target_labels)

    rng = np.random.default_rng(20261018)
    augmented_count = 0
    train_images = experiment_root / "images" / "train"
    train_labels = experiment_root / "labels" / "train"
    source_train_labels = DATASET_ROOT / "labels" / "train"
    source_train_images = DATASET_ROOT / "images" / "train"

    for label_path in sorted(source_train_labels.glob("*.txt")):
        lines = read_label_lines(label_path)
        if not contains_target_class(lines):
            continue
        image_path = find_image(source_train_images, label_path.stem)
        if image_path is None:
            raise FileNotFoundError(f"Image missing for label: {label_path.name}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        augmented = make_photometric_variant(image, rng)
        output_stem = f"{image_path.stem}_ka_aug1"
        output_image = train_images / f"{output_stem}.jpg"
        output_label = train_labels / f"{output_stem}.txt"
        if not cv2.imwrite(str(output_image), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Could not write augmented image: {output_image}")
        output_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        augmented_count += 1

    data_config = dict(original_data)
    data_config["path"] = str(experiment_root).replace("\\", "/")
    data_config["train"] = "images/train"
    data_config["val"] = "images/validation"
    data_config["test"] = "images/test"
    experiment_yaml = experiment_root / "data.yaml"
    experiment_yaml.write_text(yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return experiment_yaml, augmented_count


def print_metrics(label, metrics, names):
    print(f"\n{label}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    class_maps = metrics.box.maps
    for class_id in [17, 20]:
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        if class_id < len(class_maps):
            print(f"{class_name} class mAP50-95: {float(class_maps[class_id]):.4f}")


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}\n"
            "Place this script in the project root and verify the dataset folder path."
        )
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(
            f"Base character model not found: {BASE_MODEL}\n"
            "The script will not train from a guessed or unrelated model."
        )
    source_yaml = DATASET_ROOT / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    original_data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = DATASET_ROOT / f"ka_balanced_experiment_{timestamp}"
    experiment_root.mkdir(parents=True, exist_ok=False)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Device: {device}")
    if device == "cpu":
        print("CPU training may take a long time.")

    experiment_yaml, augmented_count = build_experiment_dataset(experiment_root, original_data)
    original_ka = count_class_objects(DATASET_ROOT / "labels" / "train", CLASS_TO_BALANCE)
    original_ha = count_class_objects(DATASET_ROOT / "labels" / "train", 17)
    balanced_ka = count_class_objects(experiment_root / "labels" / "train", CLASS_TO_BALANCE)
    balanced_ha = count_class_objects(experiment_root / "labels" / "train", 17)

    print(f"\nAdded training-only photometric variants for {CLASS_NAME_TO_BALANCE}: {augmented_count}")
    print(f"Original training counts: ka={original_ka}, ha={original_ha}")
    print(f"Experiment training counts: ka={balanced_ka}, ha={balanced_ha}")
    print("Validation and test images/labels were copied without augmentation.")
    print("The production model file will not be overwritten.")

    evaluation_root = experiment_root / "evaluation"
    baseline_model = YOLO(str(BASE_MODEL))
    baseline_metrics = baseline_model.val(
        data=str(experiment_yaml),
        split="val",
        imgsz=640,
        batch=8,
        device=device,
        plots=False,
        project=str(evaluation_root),
        name="baseline_validation",
        exist_ok=True,
        workers=0,
    )
    print_metrics("BASELINE VALIDATION", baseline_metrics, baseline_model.names)

    model = YOLO(str(BASE_MODEL))
    train_name = "ka_balanced_finetune"
    train_project = experiment_root / "runs"
    results = model.train(
        data=str(experiment_yaml),
        epochs=60,
        imgsz=640,
        batch=8,
        patience=12,
        device=device,
        workers=0,
        project=str(train_project),
        name=train_name,
        exist_ok=False,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        degrees=1.0,
        translate=0.03,
        scale=0.10,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.10,
        hsv_v=0.15,
        plots=True,
        save=True,
        verbose=True,
    )

    best_model_path = train_project / train_name / "weights" / "best.pt"
    if not best_model_path.is_file():
        trainer = getattr(model, "trainer", None)
        possible = Path(trainer.best) if trainer is not None and getattr(trainer, "best", None) else None
        if possible is not None and possible.is_file():
            best_model_path = possible
        else:
            raise FileNotFoundError("Training finished, but best.pt could not be located. Check the run folder.")

    trained_model = YOLO(str(best_model_path))
    trained_metrics = trained_model.val(
        data=str(experiment_yaml),
        split="val",
        imgsz=640,
        batch=8,
        device=device,
        plots=False,
        project=str(evaluation_root),
        name="finetuned_validation",
        exist_ok=True,
        workers=0,
    )
    print_metrics("FINE-TUNED VALIDATION", trained_metrics, trained_model.names)
    print("\nEXPERIMENT COMPLETE")
    print(f"Best experimental model: {best_model_path}")
    print(f"Experiment dataset: {experiment_root}")
    print("Do not replace models/character/character_best.pt yet. Compare the validation results first.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
