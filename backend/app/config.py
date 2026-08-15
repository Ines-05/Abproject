import os
from dataclasses import dataclass
from pathlib import Path

_VISION_MODULE_DIR = Path(__file__).resolve().parent.parent / "vision_module"
_DEFAULT_LOCALIZER_PATH = _VISION_MODULE_DIR / "best.pt"
_DEFAULT_CLASSIFIER_PATH = _VISION_MODULE_DIR / "cvtechniques_best.pt"


@dataclass(frozen=True)
class Settings:
    localizer_model_path: Path = Path(os.getenv("VISION_LOCALIZER_MODEL_PATH", str(_DEFAULT_LOCALIZER_PATH)))
    classifier_model_path: Path = Path(os.getenv("VISION_CLASSIFIER_MODEL_PATH", str(_DEFAULT_CLASSIFIER_PATH)))
    max_upload_bytes: int = int(os.getenv("VISION_MAX_UPLOAD_BYTES", 15 * 1024 * 1024))
    default_localizer_conf: float = float(os.getenv("VISION_LOCALIZER_CONF", 0.25))
    default_classifier_conf: float = float(os.getenv("VISION_CLASSIFIER_CONF", 0.05))
    default_iou: float = float(os.getenv("VISION_DEFAULT_IOU", 0.45))
    crop_padding_frac: float = float(os.getenv("VISION_CROP_PADDING_FRAC", 0.15))
    shelf_gap_factor: float = float(os.getenv("VISION_SHELF_GAP_FACTOR", 0.6))
    shelf_min_gap_px: float = float(os.getenv("VISION_SHELF_MIN_GAP_PX", 4.0))


settings = Settings()
