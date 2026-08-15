from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


@dataclass
class RawDetection:
    label: str
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2


class Detector:
    def __init__(self, model_path: Path):
        self.model = YOLO(str(model_path))

    def predict(self, image: Image.Image, conf: float, iou: float) -> list[RawDetection]:
        results = self.model.predict(source=image, conf=conf, iou=iou, verbose=False)
        return self._to_detections(results[0])

    def predict_batch(self, images: list[Image.Image], conf: float, iou: float) -> list[list[RawDetection]]:
        if not images:
            return []
        results = self.model.predict(source=images, conf=conf, iou=iou, verbose=False)
        return [self._to_detections(result) for result in results]

    @staticmethod
    def _to_detections(result) -> list[RawDetection]:
        names = result.names
        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                RawDetection(
                    label=names[class_id],
                    class_id=class_id,
                    confidence=float(box.conf.item()),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
