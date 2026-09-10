"""
src/vision/recognizer.py
Utilitário de verificação de presença de carta e extração de canto superior esquerdo.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    CARD_BRIGHTNESS_THRESHOLD,
    CARDS_DIR,
    MAX_GREEN_RATIO,
    RANKS_DIR,
    SUITS_DIR,
)


def ensure_directories():
    """Garante a existência das pastas necessárias para templates e cards."""
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    RANKS_DIR.mkdir(parents=True, exist_ok=True)
    SUITS_DIR.mkdir(parents=True, exist_ok=True)


def is_card_present(
    slot_img: np.ndarray,
    brightness_threshold: int = CARD_BRIGHTNESS_THRESHOLD,
    max_green_ratio: float = MAX_GREEN_RATIO
) -> Tuple[bool, float, float]:
    """
    Analisa se o slot contém uma carta ou apenas o feltro verde da mesa.
    """
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([85, 255, 200])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    total_pixels = slot_img.shape[0] * slot_img.shape[1]
    green_ratio = float(np.sum(green_mask > 0) / total_pixels)

    has_card = (mean_brightness >= brightness_threshold) and (green_ratio < max_green_ratio)
    return has_card, mean_brightness, green_ratio


def extract_corner(slot_img: np.ndarray, width_pct: float = 0.35, height_pct: float = 0.45) -> np.ndarray:
    """Extrai o canto superior esquerdo da carta (Rank + Suit)."""
    h, w = slot_img.shape[:2]
    corner_h = int(round(h * height_pct))
    corner_w = int(round(w * width_pct))
    return slot_img[0:corner_h, 0:corner_w]


def process_slots(num_slots: int = 5):
    ensure_directories()

    print("=" * 70)
    print("RECONHECEDOR DE SLOTS DO BOARD - Poker Analytics")
    print("=" * 70)
    print(f"Diretório dos slots: {CARDS_DIR}")
    print(f"Estrutura de templates preparada em: assets/templates/ (ranks e suits)")
    print("-" * 70)

    for i in range(1, num_slots + 1):
        slot_filename = f"slot_{i}.png"
        slot_path = CARDS_DIR / slot_filename

        if not slot_path.exists():
            print(f"Slot {i}: Arquivo '{slot_filename}' não encontrado.")
            continue

        slot_img = cv2.imread(str(slot_path))
        if slot_img is None:
            print(f"Slot {i}: Erro ao ler '{slot_filename}'.")
            continue

        has_card, brightness, green_ratio = is_card_present(slot_img)

        if has_card:
            status = "CARTA DETECTADA"
            corner_crop = extract_corner(slot_img, width_pct=0.35, height_pct=0.45)
            corner_filename = f"corner_slot_{i}.png"
            corner_path = CARDS_DIR / corner_filename
            cv2.imwrite(str(corner_path), corner_crop)
            dest_info = str(corner_path)
        else:
            status = "VAZIO (FELTRO VERDE)"
            dest_info = "Nenhum recorte gerado"

        print(f"Slot {i:02d}: [{status}]")
        print(f"   -> Brilho Médio: {brightness:.1f} | Verde na imagem: {green_ratio * 100:.1f}%")
        if has_card:
            print(f"   -> Recorte do canto salvo em: {dest_info}")
        print("-" * 70)

    print("=" * 70)
    print("Processamento concluído.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    process_slots()
