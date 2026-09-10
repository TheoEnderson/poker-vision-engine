"""
src/vision/match_small.py
Utilitário de correspondência de glifos pequenos (ícones reduzidos de naipe e rank).
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


def match_small_glyph(
    sub_region: np.ndarray,
    templates: Dict[str, np.ndarray],
    scales: Optional[np.ndarray] = None
) -> Tuple[Optional[str], float]:
    """
    Compara uma sub-região de glifo reduzido com um conjunto de templates em multi-escala.
    """
    if sub_region is None or sub_region.size == 0 or not templates:
        return None, -1.0

    if scales is None:
        scales = np.linspace(0.6, 1.5, 15)

    best_match = None
    best_score = -1.0

    for name, tmpl in templates.items():
        for scale in scales:
            sw = int(tmpl.shape[1] * scale)
            sh = int(tmpl.shape[0] * scale)
            if sw == 0 or sh == 0 or sw > sub_region.shape[1] or sh > sub_region.shape[0]:
                continue
            resized = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(sub_region, resized, cv2.TM_CCOEFF_NORMED)
            score = float(np.max(res))
            if score > best_score:
                best_score = score
                best_match = name

    return best_match, best_score
