from dataclasses import dataclass

from PIL import Image

from app.inference import Detector


@dataclass
class CombinedDetection:
    label: str
    class_id: int
    bbox_confidence: float
    label_confidence: float
    bbox: tuple[float, float, float, float]


def padded_box(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    pad_frac: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    pad_x = width * pad_frac
    pad_y = height * pad_frac
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(img_w), x2 + pad_x),
        min(float(img_h), y2 + pad_y),
    )


def detect_and_classify(
    image: Image.Image,
    localizer: Detector,
    classifier: Detector,
    localizer_conf: float,
    classifier_conf: float,
    iou: float,
    pad_frac: float,
) -> list[CombinedDetection]:
    boxes = localizer.predict(image, conf=localizer_conf, iou=iou)
    if not boxes:
        return []

    crops = []
    for box in boxes:
        crop_bbox = padded_box(box.bbox, image.width, image.height, pad_frac)
        crops.append(image.crop(crop_bbox))

    classifications = classifier.predict_batch(crops, conf=classifier_conf, iou=iou)

    combined = []
    for box, crop_detections in zip(boxes, classifications):
        if crop_detections:
            best = max(crop_detections, key=lambda d: d.confidence)
            label, class_id, label_confidence = best.label, best.class_id, best.confidence
        else:
            label, class_id, label_confidence = "unknown", -1, 0.0

        combined.append(
            CombinedDetection(
                label=label,
                class_id=class_id,
                bbox_confidence=box.confidence,
                label_confidence=label_confidence,
                bbox=box.bbox,
            )
        )
    return combined
