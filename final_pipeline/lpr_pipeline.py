# Bangladesh License Plate Recognition Pipeline

from PIL import Image
import numpy as np
from ultralytics import YOLO


def boxes_overlap_or_close(a, b, gap_ratio=0.08):
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b

    wa = x2a - x1a
    wb = x2b - x1b
    gap = max(wa, wb) * gap_ratio

    return not (
        x2a + gap < x1b or
        x2b + gap < x1a or
        y2a + gap < y1b or
        y2b + gap < y1a
    )


def merge_boxes(boxes):
    boxes = [list(b) for b in boxes]
    changed = True

    while changed:
        changed = False
        result = []

        while boxes:
            current = boxes.pop(0)
            merged = False

            for i, other in enumerate(boxes):
                if boxes_overlap_or_close(current, other):
                    current = [
                        min(current[0], other[0]),
                        min(current[1], other[1]),
                        max(current[2], other[2]),
                        max(current[3], other[3])
                    ]

                    boxes.pop(i)
                    boxes.insert(0, current)
                    merged = True
                    changed = True
                    break

            if not merged:
                result.append(current)

        boxes = result

    return boxes


def remove_character_duplicates(characters, distance_ratio=0.35):
    if not characters:
        return []

    chars = sorted(
        characters,
        key=lambda c: c["confidence"],
        reverse=True
    )

    kept = []

    for char in chars:
        duplicate = False

        for existing in kept:
            avg_width = (char["w"] + existing["w"]) / 2
            avg_height = (char["h"] + existing["h"]) / 2

            x_close = (
                abs(char["x"] - existing["x"])
                < avg_width * distance_ratio
            )

            y_close = (
                abs(char["y"] - existing["y"])
                < avg_height * distance_ratio
            )

            if x_close and y_close:
                duplicate = True
                break

        if not duplicate:
            kept.append(char)

    return kept


def reconstruct_plate_text(characters):
    if not characters:
        return None

    if len(characters) == 1:
        return characters[0]["label"]

    chars_y = sorted(characters, key=lambda c: c["y"])

    best_gap = -1
    split_index = None

    for i in range(len(chars_y) - 1):
        gap = chars_y[i + 1]["y"] - chars_y[i]["y"]

        if gap > best_gap:
            best_gap = gap
            split_index = i + 1

    top_row = chars_y[:split_index]
    bottom_row = chars_y[split_index:]

    top_row.sort(key=lambda c: c["x"])
    bottom_row.sort(key=lambda c: c["x"])

    top_text = " ".join(c["label"] for c in top_row)
    bottom_text = "".join(c["label"] for c in bottom_row)

    return f"{top_text}\n{bottom_text}"


class BangladeshLPR:

    def __init__(self, localization_model_path, character_model_path):
        self.localization_model = YOLO(str(localization_model_path))
        self.character_model = YOLO(str(character_model_path))

    def recognize(self, image_path):
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        loc_result = self.localization_model.predict(
            source=np.array(image),
            conf=0.05,
            iou=0.5,
            imgsz=640,
            device="cpu",
            verbose=False
        )[0]

        if len(loc_result.boxes) == 0:
            return {
                "status": "no_plate_detected",
                "text": None
            }

        detections = []

        for box in loc_result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": confidence
            })

        merged_boxes = merge_boxes(
            [d["box"] for d in detections]
        )

        best_box = max(
            merged_boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )

        x1, y1, x2, y2 = best_box

        padding = (x2 - x1) * 0.12

        x1 = max(0, int(x1 - padding))
        x2 = min(img_w, int(x2 + padding))
        y1 = max(0, int(y1))
        y2 = min(img_h, int(y2))

        plate_crop = image.crop((x1, y1, x2, y2))


        char_result = self.character_model.predict(
            source=plate_crop,
            conf=0.10,
            iou=0.5,
            imgsz=640,
            device="cpu",
            verbose=False
        )[0]

        characters = []

        for box in char_result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            cx, cy, w, h = map(
                float,
                box.xywh[0].tolist()
            )

            characters.append({
                "label": self.character_model.names[cls_id],
                "confidence": confidence,
                "x": cx,
                "y": cy,
                "w": w,
                "h": h
            })

        characters = remove_character_duplicates(characters)

        if not characters:
            return {
                "status": "plate_found_no_characters",
                "text": None,
                "crop": plate_crop,
                "localization_detections": detections
            }

        final_text = reconstruct_plate_text(characters)

        return {
            "status": "success",
            "text": final_text,
            "crop": plate_crop,
            "plate_box": [x1, y1, x2, y2],
            "localization_detections": detections,
            "characters": characters
        }
