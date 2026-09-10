"""
src/vision/match_contours.py
Utilitários de segmentação e correspondência por contornos para glifos e símbolos.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


def find_glyph_bounding_boxes(
    binary_img: np.ndarray,
    min_area: float = 10.0,
    min_w: int = 5,
    min_h: int = 5
) -> List[Tuple[int, int, int, int]]:
    """
    Encontra caixas delimitadoras de contornos externos válidos em imagem binarizada.
    Retorna lista de (x, y, w, h) ordenada decrescente por área.
    """
    if binary_img is None or binary_img.size == 0:
        return []

    cnts, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area >= min_area and w >= min_w and h >= min_h:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def match_dominant_contour(
    region_bin: np.ndarray,
    templates: Dict[str, np.ndarray],
    scales: Optional[np.ndarray] = None
) -> Tuple[Optional[str], float]:
    """
    Localiza o contorno dominante na ROI e executa multiscale template matching.
    """
    boxes = find_glyph_bounding_boxes(region_bin)
    if not boxes:
        return None, -1.0

    hx, hy, hw, hh = boxes[0]
    glyph = region_bin[hy : hy + hh, hx : hx + hw]

    if scales is None:
        scales = np.linspace(0.6, 1.5, 15)

    best_name = None
    best_score = -1.0

    for name, tmpl in templates.items():
        for s in scales:
            sw = int(tmpl.shape[1] * s)
            sh = int(tmpl.shape[0] * s)
            if sw == 0 or sh == 0 or sw > glyph.shape[1] or sh > glyph.shape[0]:
                continue
            resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(glyph, resized, cv2.TM_CCOEFF_NORMED)
            score = float(np.max(res))
            if score > best_score:
                best_score = score
                best_name = name

    return best_name, best_score
