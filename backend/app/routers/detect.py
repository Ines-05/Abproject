import io
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.inference import Detector
from app.pipeline import detect_and_classify
from app.schemas import BBox, Detection, DetectResponse
from app.shelf_clustering import assign_shelves

logger = logging.getLogger(__name__)
router = APIRouter()


def get_localizer(request: Request) -> Detector:
    return request.app.state.localizer


def get_classifier(request: Request) -> Detector:
    return request.app.state.classifier


@router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile,
    localizer_conf: float = Query(default=settings.default_localizer_conf, ge=0.0, le=1.0),
    classifier_conf: float = Query(default=settings.default_classifier_conf, ge=0.0, le=1.0),
    iou: float = Query(default=settings.default_iou, ge=0.0, le=1.0),
    localizer: Detector = Depends(get_localizer),
    classifier: Detector = Depends(get_classifier),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    try:
        combined_detections = detect_and_classify(
            image,
            localizer=localizer,
            classifier=classifier,
            localizer_conf=localizer_conf,
            classifier_conf=classifier_conf,
            iou=iou,
            pad_frac=settings.crop_padding_frac,
        )
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")

    shelf_ids, shelf_count = assign_shelves(
        combined_detections,
        gap_factor=settings.shelf_gap_factor,
        min_gap_px=settings.shelf_min_gap_px,
    )

    detections = [
        Detection(
            label=d.label,
            class_id=d.class_id,
            bbox_confidence=d.bbox_confidence,
            label_confidence=d.label_confidence,
            bbox=BBox(x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3]),
            shelf=shelf,
        )
        for d, shelf in zip(combined_detections, shelf_ids)
    ]
    detections.sort(key=lambda d: (d.shelf, d.bbox.x1))

    return DetectResponse(
        image_width=image.width,
        image_height=image.height,
        shelf_rows_found=shelf_count,
        detections=detections,
    )
