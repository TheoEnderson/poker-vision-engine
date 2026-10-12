"""Identidade explícita da rodada Replay Poker; independente do OCR financeiro."""

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pytesseract

from .ocr_reader import _resolve_image_input


def _parse_hand_identity(text: str) -> Optional[Tuple[str, str]]:
    """Exige os dois rótulos e números integrais, sem corrigir caracteres OCR."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) != 2:
        return None
    current = re.fullmatch(r"(?:Atual|Current):\s*#([0-9]+)", lines[0], re.IGNORECASE)
    previous = re.fullmatch(r"(?:Anterior|Previous):\s*#([0-9]+)", lines[1], re.IGNORECASE)
    if not current or not previous:
        return None
    ids = current[1], previous[1]
    if int(ids[0]) <= int(ids[1]) or int(ids[1]) <= 0:
        return None
    return ids


def extract_hand_identity(image: Union[str, Path, np.ndarray], diagnostics=None) -> Optional[Tuple[str, str]]:
    """Lê o painel superior esquerdo no layout 1920x1080 existente.

    Exige concordância exata entre texto nativo e ampliado, incluindo rótulos
    e '#'. Uma leitura incompleta/divergente NÃO é evidência de nova mão.
    Não há fallback para valores antigos nem reconstrução de dígitos.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(readings=[], status="UNKNOWN")
    if not isinstance(image, (str, Path, np.ndarray)):
        return None
    frame = _resolve_image_input(image)
    if frame is None or frame.shape[:2] != (1080, 1920):
        return None
    gray = cv2.cvtColor(frame[147:194, 0:200], cv2.COLOR_BGR2GRAY)
    native_text = pytesseract.image_to_string(gray, config="--psm 6") or ""
    native = _parse_hand_identity(native_text)
    if diagnostics is not None:
        diagnostics["readings"].append(dict(method="native", text=native_text, identity=native))
    if native is None:
        return None
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    scaled_text = pytesseract.image_to_string(enlarged, config="--psm 6") or ""
    scaled = _parse_hand_identity(scaled_text)
    if diagnostics is not None:
        diagnostics["readings"].append(dict(method="scale3", text=scaled_text, identity=scaled))
        diagnostics["status"] = "CONFIRMED" if native == scaled else "CONFLICT"
    return native if native == scaled else None
