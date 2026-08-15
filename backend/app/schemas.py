from pydantic import BaseModel


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    class_id: int
    bbox_confidence: float
    label_confidence: float
    bbox: BBox
    shelf: int


class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    shelf_rows_found: int
    detections: list[Detection]
