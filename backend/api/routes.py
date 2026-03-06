import torch
from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from core.config import IMAGE_SIZE, CHECKPOINT_MAP
from services import model_service
from utils.image_utils import (
    decode_image_from_bytes,
    preprocess,
    tensor_to_mask_logits,
    mask_logits_to_uint8,
    mask_to_png_base64,
)

router = APIRouter()

VALID_MODELS = list(CHECKPOINT_MAP.keys())


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("Kvasir-Seg"),
):
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {VALID_MODELS}")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file")
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        image = decode_image_from_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    tensor = preprocess(image)
    logits = model_service.predict(tensor, model)
    probs = torch.sigmoid(logits)
    pred = tensor_to_mask_logits(probs)
    mask = mask_logits_to_uint8(pred, threshold=0.5)
    mask_b64 = mask_to_png_base64(mask)
    return {"mask": mask_b64, "size": list(IMAGE_SIZE), "model": model}
