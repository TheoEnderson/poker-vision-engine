"""
src/vision/build_templates.py
Utilitário para extração de templates de Ranks e Naipes a partir de recortes de cartas.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from src.config import CARDS_DIR, RANKS_DIR, SUITS_DIR


def ensure_template_dirs() -> Tuple[Path, Path]:
    """Garante que as pastas de templates de ranks e suits existam."""
    RANKS_DIR.mkdir(parents=True, exist_ok=True)
    SUITS_DIR.mkdir(parents=True, exist_ok=True)
    return RANKS_DIR, SUITS_DIR


def binarize_corner(corner_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica binarização isolando os glifos do Rank e Naipe sem ruído.
    Filtra o feltro verde da mesa e binariza os símbolos sobre o fundo branco da carta.
    """
    hsv = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)

    green_mask = cv2.inRange(hsv, np.array([30, 50, 30]), np.array([90, 255, 255]))
    glyph_mask = np.zeros_like(gray)
    glyph_mask[(green_mask == 0) & (gray < 185)] = 255
    return glyph_mask


def find_split_line(glyph_mask: np.ndarray) -> int:
    """
    Encontra a melhor linha horizontal divisória entre o Rank e o Naipe.
    """
    h = glyph_mask.shape[0]
    min_row = int(h * 0.45)
    max_row = int(h * 0.75)

    row_sums = glyph_mask.sum(axis=1)
    split_y = min_row + int(np.argmin(row_sums[min_row:max_row]))
    return split_y


def crop_to_glyph(binary_img: np.ndarray) -> np.ndarray:
    """Recorta a imagem binária na caixa delimitadora exata dos pixels brancos (255)."""
    pts = cv2.findNonZero(binary_img)
    if pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        return binary_img[y : y + h, x : x + w]
    return binary_img


def extract_and_save_templates(cards_dir: Optional[Union[str, Path]] = None):
    c_dir = Path(cards_dir) if cards_dir else CARDS_DIR
    r_dir, s_dir = ensure_template_dirs()

    print("=" * 70)
    print("CONSTRUTOR DE TEMPLATES DE RANKS E NAIPES - Poker Analytics")
    print("=" * 70)
    print(f"Diretório dos cantos: {c_dir}")
    print(f"Diretório de Ranks:   {r_dir}")
    print(f"Diretório de Naipes:  {s_dir}")
    print("-" * 70)

    corner_files = sorted(c_dir.glob("corner_slot_*.png"))
    if not corner_files:
        print("[AVISO] Nenhum arquivo 'corner_slot_*.png' encontrado.")
        return

    for corner_path in corner_files:
        corner_bgr = cv2.imread(str(corner_path))
        if corner_bgr is None:
            continue

        glyph_mask = binarize_corner(corner_bgr)
        split_y = find_split_line(glyph_mask)

        rank_crop = crop_to_glyph(glyph_mask[:split_y, :])
        suit_crop = crop_to_glyph(glyph_mask[split_y:, :])

        print(f"Processando {corner_path.name}:")
        print(f"  -> Rank: {rank_crop.shape[1]}x{rank_crop.shape[0]} px")
        print(f"  -> Naipe: {suit_crop.shape[1]}x{suit_crop.shape[0]} px")

    print("=" * 70)


if __name__ == "__main__":
    extract_and_save_templates()
