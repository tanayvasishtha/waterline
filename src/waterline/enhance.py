"""Enhancement operators the agent can ask for on a re-read."""

import cv2
import numpy as np


def clahe(image: np.ndarray, clip: float = 3.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    return np.asarray(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def mask_glare(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 240) & (hsv[:, :, 1] < 40)).astype(np.uint8) * 255
    mask = np.asarray(cv2.dilate(mask, np.ones((5, 5), np.uint8)))
    return np.asarray(cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA))


def gamma(image: np.ndarray, g: float = 1.4) -> np.ndarray:
    table = (np.linspace(0, 1, 256) ** (1 / g) * 255).astype(np.uint8)
    return np.asarray(cv2.LUT(image, table))


def apply_op(image: np.ndarray, op: str) -> np.ndarray:
    if op == "clahe":
        return clahe(image)
    if op == "mask_glare":
        return mask_glare(image)
    if op == "gamma":
        return gamma(image)
    raise ValueError(f"unknown enhance op: {op}")
