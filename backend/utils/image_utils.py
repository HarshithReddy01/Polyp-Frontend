import base64
import numpy as np
import cv2
import torch

from core.config import IMAGE_SIZE, DEVICE


def decode_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def preprocess(image: np.ndarray) -> torch.Tensor:
    image = cv2.resize(image, IMAGE_SIZE)
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    tensor = torch.from_numpy(image).to(DEVICE)
    return tensor


def tensor_to_mask_logits(y: torch.Tensor) -> np.ndarray:
    pred = y[0].detach().cpu().numpy()
    pred = np.squeeze(pred, axis=0)
    return pred


def mask_logits_to_uint8(pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    mask = (pred > threshold).astype(np.int32) * 255
    return np.array(mask, dtype=np.uint8)


def mask_to_png_base64(mask: np.ndarray) -> str:
    success, buf = cv2.imencode(".png", mask)
    if not success:
        raise ValueError("Failed to encode mask as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")
