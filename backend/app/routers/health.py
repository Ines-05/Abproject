from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "localizer_loaded": getattr(request.app.state, "localizer", None) is not None,
        "classifier_loaded": getattr(request.app.state, "classifier", None) is not None,
    }
