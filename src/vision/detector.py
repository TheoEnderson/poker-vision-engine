"""
src/vision/detector.py
Módulo de Visão Computacional para detecção de cartas do board, mão do Hero e contagem de oponentes.
"""

import datetime
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    CARD_BRIGHTNESS_THRESHOLD,
    DEFAULT_RANK_THRESHOLD,
    DEFAULT_SUIT_THRESHOLD,
    MAX_GREEN_RATIO,
    PROJECT_ROOT,
    RANKS_DIR,
    SAMPLES_DIR,
    SUITS_DIR,
    UNKNOWN_CARDS_DIR,
)


def _resolve_image_input(image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    """Resolve entrada de imagem permitindo arrays numpy, caminhos absolutos ou relativos (com fallback em samples)."""
    if isinstance(image_input, np.ndarray):
        return image_input

    path_obj = Path(image_input)
    if path_obj.is_absolute() and path_obj.exists():
        return cv2.imread(str(path_obj))

    # Verifica no CWD
    if path_obj.exists():
        return cv2.imread(str(path_obj))

    # Verifica em assets/samples/
    sample_path = SAMPLES_DIR / path_obj.name
    if sample_path.exists():
        return cv2.imread(str(sample_path))

    # Verifica em PROJECT_ROOT
    root_path = PROJECT_ROOT / path_obj.name
    if root_path.exists():
        return cv2.imread(str(root_path))

    return None


def save_unknown_card(card_crop: np.ndarray, prefix: str = "card", verbose: bool = True) -> str:
    """
    Salva automaticamente um recorte de carta não identificada em assets/unknown_cards/
    com timestamp, permitindo rápido catalogamento de templates faltantes.
    """
    UNKNOWN_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"unknown_{prefix}_{timestamp}.png"
    filepath = UNKNOWN_CARDS_DIR / filename

    try:
        cv2.imwrite(str(filepath), card_crop)
        if verbose:
            print(f"[AUTO-SAVE] Carta não identificada salva em: assets/unknown_cards/{filename}")
    except Exception as e:
        if verbose:
            print(f"[ERRO AUTO-SAVE] Falha ao salvar carta desconhecida: {e}")

    return str(filepath)


def crop_to_glyph(binary_img: np.ndarray) -> np.ndarray:
    """Recorta a imagem binária na caixa delimitadora exata dos pixels ativos (255)."""
    pts = cv2.findNonZero(binary_img)
    if pts is not None:
        x, y, w, h = cv2.boundingRect(pts)
        return binary_img[y : y + h, x : x + w]
    return binary_img


def load_templates(
    ranks_dir: Optional[Union[str, Path]] = None,
    suits_dir: Optional[Union[str, Path]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Carrega todos os templates de ranks e suits disponíveis.
    Recorta o glifo na sua caixa delimitadora (bounding box) para permitir
    deslocamento horizontal/vertical livre no matchTemplate.
    """
    r_dir = Path(ranks_dir) if ranks_dir else RANKS_DIR
    s_dir = Path(suits_dir) if suits_dir else SUITS_DIR

    ranks: Dict[str, np.ndarray] = {}
    for path in sorted(r_dir.glob("*.png")):
        name = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ranks[name] = crop_to_glyph(img)

    suits: Dict[str, np.ndarray] = {}
    for path in sorted(s_dir.glob("*.png")):
        name = path.stem
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            suits[name] = crop_to_glyph(img)

    return ranks, suits


def is_card_present(
    slot_img: np.ndarray,
    brightness_threshold: int = CARD_BRIGHTNESS_THRESHOLD,
    max_green_ratio: float = MAX_GREEN_RATIO
) -> bool:
    """Verifica se há uma carta no slot ou apenas o feltro verde da mesa."""
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    hsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 60, 40])
    upper_green = np.array([85, 255, 200])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = slot_img.shape[0] * slot_img.shape[1]
    green_ratio = float(np.sum(green_mask > 0) / total_pixels)

    return (mean_brightness >= brightness_threshold) and (green_ratio < max_green_ratio)


def binarize_corner(corner_bgr: np.ndarray) -> np.ndarray:
    """Binariza o canto da carta isolando os símbolos (glifos) de Rank e Naipe sem ruído."""
    hsv = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)

    # Remove o feltro verde do fundo
    green_mask = cv2.inRange(hsv, np.array([30, 50, 30]), np.array([90, 255, 255]))

    glyph_mask = np.zeros_like(gray)
    glyph_mask[(green_mask == 0) & (gray < 185)] = 255
    return glyph_mask


def resolve_red_suit_tie(roi_thresh: np.ndarray) -> str:
    """
    Desempate geométrico entre Ouros ('d') e Copas ('h').
    Copas possui um vale central no topo (dois lóbulos).
    Ouros possui um vértice agudo (um único pico).
    """
    cnts, _ = cv2.findContours(roi_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 'h'
        
    valid_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 15 and w >= 5 and h >= 5:
            valid_cnts.append((x*x + y*y, (x, y, w, h), c))
            
    if not valid_cnts:
        return 'h'
        
    valid_cnts.sort(key=lambda x: x[0])
    _, (x, y, w, h), c = valid_cnts[0]
    glyph_crop = roi_thresh[y : y + h, x : x + w]
    
    third_h = max(2, h // 3)
    for r in range(third_h):
        row_pixels = glyph_crop[r]
        cols = np.where(row_pixels > 0)[0]
        if len(cols) > 0:
            left, right = cols[0], cols[-1]
            width = right - left + 1
            if width > 4 and len(cols) < width - 1:
                return 'h'
                
    return 'd'


def resolve_black_suit_tie(roi_thresh: np.ndarray) -> str:
    """
    Desempate morfológico entre Paus ('c') e Espadas ('s').
    O Paus (Club) possui lóbulos arredondados, causando um 'pulo' na largura 
    (contração seguida de expansão). A Espada (Spade) expande continuamente.
    """
    cnts, _ = cv2.findContours(roi_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 's'
        
    valid_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 15 and w >= 5 and h >= 5:
            valid_cnts.append((x*x + y*y, (x, y, w, h), c))
            
    if not valid_cnts:
        return 's'
        
    valid_cnts.sort(key=lambda x: x[0])
    _, (x, y, w, h), c = valid_cnts[0]
    glyph_crop = roi_thresh[y : y + h, x : x + w]
    
    half_h = max(2, int(h * 0.6))
    widths = []
    for r in range(half_h):
        cols = np.where(glyph_crop[r] > 0)[0]
        if len(cols) > 0:
            widths.append(cols[-1] - cols[0] + 1)
        else:
            widths.append(0)
            
    if len(widths) < 2:
        return 's'
        
    diffs = np.diff(widths)
    if np.any(diffs >= 4):
        return 'c'
        
    max_w = 0
    for w_val in widths:
        if w_val > max_w:
            max_w = w_val
        elif w_val < max_w:
            return 'c'
            
    return 's'


def match_glyph_multiscale(
    roi: np.ndarray,
    templates: Dict[str, np.ndarray],
    scales: Optional[Union[List[float], np.ndarray]] = None
) -> Tuple[Optional[str], float]:
    """
    Realiza Template Matching multi-escala em uma ROI contra um conjunto de templates.
    Testa de 60% a 150% do tamanho em 25 passos para ser imune a diferenças de escala.
    """
    if scales is None:
        scales = np.linspace(0.6, 1.5, 25)

    best_score = -1.0
    best_match = None

    for template_name, template_img in templates.items():
        for scale in [1.0] + list(scales):
            width = int(template_img.shape[1] * scale)
            height = int(template_img.shape[0] * scale)
            if width == 0 or height == 0:
                continue

            resized_template = cv2.resize(template_img, (width, height), interpolation=cv2.INTER_AREA)

            # Garante que o template cabe dentro da ROI
            if resized_template.shape[0] > roi.shape[0] or resized_template.shape[1] > roi.shape[1]:
                continue

            res = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_match = template_name

    return best_match, best_score


def filter_suits_by_color(bgr_suit_crop: np.ndarray, suits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Verifica a cor predominante do recorte do naipe (usando a máscara HSV de vermelho).
    Se houver saturação de vermelho significativa (>15% dos pixels do glifo):
      restringe a busca aos naipes vermelhos: ['h', 'd'].
    Caso contrário:
      restringe a busca aos naipes pretos: ['c', 's'].
    """
    hsv = cv2.cvtColor(bgr_suit_crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_suit_crop, cv2.COLOR_BGR2GRAY)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    red_mask = ((h < 15) | (h > 165)) & (s > 80) & (v > 70)

    green_mask = cv2.inRange(hsv, np.array([30, 50, 30]), np.array([90, 255, 255]))
    glyph_mask = (green_mask == 0) & (gray < 190)

    total_glyph = np.sum(glyph_mask)
    total_red = np.sum(red_mask & glyph_mask)

    if total_glyph > 0 and (total_red / total_glyph) > 0.15:
        target_suits = {"h", "d"}
    else:
        target_suits = {"c", "s"}

    return {k: v for k, v in suits.items() if k in target_suits}


def detect_board(
    board_input: Union[str, Path, np.ndarray] = "mesa_6.png",
    threshold: float = DEFAULT_RANK_THRESHOLD,
    verbose: bool = True,
    num_slots: int = 5,
    board_coords: Optional[Dict[str, int]] = None
) -> List[str]:
    """Detecta as 5 cartas do board comunitário na imagem da mesa."""
    ranks, suits = load_templates()

    if verbose:
        print("=" * 70)
        print("DETECTOR DE CARTAS DO BOARD - Poker Analytics")
        print("=" * 70)
        print(f"Templates de Rank carregados ({len(ranks)}): {', '.join(ranks.keys())}")
        print(f"Templates de Naipe carregados ({len(suits)}): {', '.join(suits.keys())}")
        print(f"Limiar de confiança (score mínimo): {threshold}")
        print("-" * 70)

    board = _resolve_image_input(board_input)
    if board is None:
        if verbose:
            print(f"[ERRO] Não foi possível carregar o board de: {board_input}")
        return []

    height, width = board.shape[:2]

    # Se for uma imagem da mesa completa (ex: 1920x1080), recorta a região do board
    if board_coords is not None:
        bt, bl = board_coords.get("top", 0), board_coords.get("left", 0)
        bw, bh = board_coords.get("width", width), board_coords.get("height", height)
        board = board[bt : bt + bh, bl : bl + bw]
        height, width = board.shape[:2]
    elif width > 600 and height > 400:
        bt, bl, bw, bh = 415, 734, 449, 109
        board = board[bt : bt + bh, bl : bl + bw]
        height, width = board.shape[:2]

    detected_board = []

    # Fatiar os 5 slots
    for i in range(num_slots):
        x_start = int(round(i * width / float(num_slots)))
        x_end = int(round((i + 1) * width / float(num_slots)))
        slot_img = board[0:height, x_start:x_end]

        if not is_card_present(slot_img):
            detected_board.append("Vazio")
            if verbose:
                print(f"Slot {i + 1}: [Vazio / Feltro da mesa]")
            continue

        corner_h = min(height, max(60, int(round(height * 0.55))))
        corner_w = int(round(slot_img.shape[1] * 0.35))
        corner = slot_img[0:corner_h, 0:corner_w]

        glyph_mask = binarize_corner(corner)

        rank_region = glyph_mask[:36, :]
        suit_region = glyph_mask[20:, :]

        best_rank, best_rank_score = match_glyph_multiscale(rank_region, ranks)

        candidate_suits = filter_suits_by_color(corner[20:, :], suits)
        best_suit, best_suit_score = match_glyph_multiscale(suit_region, candidate_suits)

        rank_ok = (best_rank is not None) and (best_rank_score >= threshold)
        suit_ok = (best_suit is not None) and (best_suit_score >= threshold)

        if rank_ok and suit_ok:
            card_str = f"{best_rank}{best_suit}"
            if verbose:
                print(
                    f"Slot {i + 1}: Rank='{best_rank}' ({best_rank_score:.3f}) | "
                    f"Naipe='{best_suit}' ({best_suit_score:.3f}) -> {card_str}"
                )
        else:
            card_str = "Desconhecida"
            save_unknown_card(slot_img, prefix=f"board_slot_{i + 1}", verbose=verbose)
            if verbose:
                print(
                    f"Slot {i + 1}: Confiança insuficiente (Rank: {best_rank_score:.3f}, "
                    f"Naipe: {best_suit_score:.3f}) -> {card_str}"
                )

        detected_board.append(card_str)

    if verbose:
        print("-" * 70)
        print(f"Board detectado: {detected_board}")
        print("=" * 70 + "\n")

    return detected_board


def find_and_detect_hero_cards(
    image_path_or_frame: Union[str, Path, np.ndarray] = "mesa_6.png",
    board_coords: Optional[Dict[str, int]] = None,
    threshold: float = DEFAULT_RANK_THRESHOLD,
    verbose: bool = True
) -> List[str]:
    """
    Localiza dinamicamente as duas cartas abertas do Hero em qualquer posição da mesa
    (6-max ou 9-max), ignorando as cartas comunitárias do board.
    """
    if board_coords is None:
        board_coords = {"top": 415, "left": 734, "width": 449, "height": 109}

    ranks, suits = load_templates()

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        if verbose:
            print(f"[ERRO] Não foi possível carregar a imagem de: {image_path_or_frame}")
        return []

    h, w = img.shape[:2]
    if w < 600 or h < 400:
        if verbose:
            print("[INFO] Imagem fornecida é um recorte do board, não uma mesa completa com assentos.")
        return []

    bt = board_coords.get("top", 415)
    bl = board_coords.get("left", 734)
    bw = board_coords.get("width", 449)
    bh = board_coords.get("height", 109)
    margin = 15

    masked = img.copy()
    cv2.rectangle(
        masked,
        (max(0, bl - margin), max(0, bt - margin)),
        (min(w, bl + bw + margin), min(h, bt + bh + margin)),
        (0, 0, 0),
        -1,
    )

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
    white_u8 = (white_mask.astype(np.uint8)) * 255

    cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        cbx, cby, cbw, cbh = cv2.boundingRect(c)
        if cby > 850 or cbx < 80 or cbx > w - 80:
            continue
        if 1500 < area < 10000 and 40 < cbw < 120 and 40 < cbh < 120:
            candidates.append((cbx, cby, cbw, cbh))
        elif 3000 < area < 20000 and 120 < cbw < 250 and 40 < cbh < 120:
            candidates.append((cbx, cby, cbw, cbh))

    hero_box = None
    if len(candidates) == 2:
        c1, c2 = sorted(candidates, key=lambda b: b[0])
        if abs(c1[1] - c2[1]) < 30 and abs((c1[0] + c1[2]) - c2[0]) < 50:
            hx = c1[0]
            hy = min(c1[1], c2[1])
            hw = (c2[0] + c2[2]) - hx
            hh = max(c1[1] + c1[3], c2[1] + c2[3]) - hy
            hero_box = (hx, hy, hw, hh)
    elif len(candidates) == 1 and candidates[0][2] >= 120:
        hero_box = candidates[0]
    elif len(candidates) > 2:
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = sorted([candidates[i], candidates[j]], key=lambda b: b[0])
                if abs(c1[1] - c2[1]) < 30 and abs((c1[0] + c1[2]) - c2[0]) < 50:
                    hx = c1[0]
                    hy = min(c1[1], c2[1])
                    hw = (c2[0] + c2[2]) - hx
                    hh = max(c1[1] + c1[3], c2[1] + c2[3]) - hy
                    hero_box = (hx, hy, hw, hh)
                    break
            if hero_box:
                break

    if not hero_box:
        if verbose:
            print("[INFO] Nenhuma carta aberta do Hero detectada fora do board (Fold ou mesa vazia).")
        return []

    hx, hy, hw, hh = hero_box
    hero_coords_dict = {"top": int(hy), "left": int(hx), "width": int(hw), "height": int(hh)}

    hero_crop = img[hy : hy + hh, hx : hx + hw]
    half_w = hw // 2
    slot1 = hero_crop[:, :half_w]
    slot2 = hero_crop[:, half_w:]

    def match_hero_slot(slot_img: np.ndarray, is_second_slot: bool = False):
        sgray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
        shsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)

        black_mask = (sgray < 185)
        h, s, v = shsv[:, :, 0], shsv[:, :, 1], shsv[:, :, 2]
        red_mask = ((h < 15) | (h > 165)) & (s > 90) & (v > 80)
        sbin = ((black_mask | red_mask).astype(np.uint8)) * 255

        rank_reg = sbin[:36, :]
        best_r, best_r_score = match_glyph_multiscale(rank_reg, ranks)

        candidate_suits = filter_suits_by_color(slot_img[16:, :35], suits)
        suit_reg = sbin[16:, :35]
        best_s, best_s_score = match_glyph_multiscale(suit_reg, candidate_suits)

        card_str = (
            f"{best_r}{best_s}"
            if (best_r_score >= threshold and best_s_score >= threshold)
            else "Desconhecida"
        )
        return card_str, best_r, best_r_score, best_s, best_s_score

    c1_str, r1, sr1, s1, ss1 = match_hero_slot(slot1, is_second_slot=False)
    c2_str, r2, sr2, s2, ss2 = match_hero_slot(slot2, is_second_slot=True)

    if c1_str == "Desconhecida":
        save_unknown_card(slot1, prefix="hero_slot_1", verbose=verbose)
    if c2_str == "Desconhecida":
        save_unknown_card(slot2, prefix="hero_slot_2", verbose=verbose)

    if c1_str == "Desconhecida" or c2_str == "Desconhecida":
        if verbose:
            print(f"[INFO] Cartas do Hero não reconhecidas com confiança suficiente: [{c1_str}, {c2_str}]. Retornando [].")
        return []

    hero_cards = [c1_str, c2_str]

    if verbose:
        print("=" * 70)
        print("DETECÇÃO DINÂMICA DA MÃO DO HERO - Poker Analytics")
        print("=" * 70)
        print("Localização encontrada na mesa (Bounding Box):")
        print(f"  {hero_coords_dict}")
        print("-" * 70)
        print(f"Slot 1: Rank='{r1}' ({sr1:.3f}) | Naipe='{s1}' ({ss1:.3f}) -> {c1_str}")
        print(f"Slot 2: Rank='{r2}' ({sr2:.3f}) | Naipe='{s2}' ({ss2:.3f}) -> {c2_str}")
        print("-" * 70)
        print(f"Cartas do Hero identificadas: {hero_cards}")
        print("=" * 70 + "\n")

    return hero_cards


def detect_active_opponents(
    image_path_or_frame: Union[str, Path, np.ndarray],
    verbose: bool = False
) -> int:
    """
    Estima o número de oponentes ativos na mão (2 a 5) contando os clusters
    de dorsos vermelhos das cartas não reveladas.
    """
    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        return 4

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if 150 < area < 5000 and 15 < w < 120 and 15 < h < 120:
            centers.append((x + w // 2, y + h // 2))

    # Agrupa cartas próximas do mesmo jogador
    opponents = 0
    grouped = []
    for cx, cy in centers:
        found = False
        for gx, gy in grouped:
            if abs(cx - gx) < 60 and abs(cy - gy) < 60:
                found = True
                break
        if not found:
            grouped.append((cx, cy))
            opponents += 1

    # Remove falsos positivos da interface, board ou cartas do Hero
    valid_opponents = 0
    for gx, gy in grouped:
        if gy > 800:
            continue  # Action buttons ou Hero's cards
        if 350 < gy < 600 and 600 < gx < 1300:
            continue  # Board area
        valid_opponents += 1

    if valid_opponents == 0:
        valid_opponents = 4

    # Numa mesa 6-max, se tirarmos o Hero, no máximo 5 oponentes
    return min(valid_opponents, 5)


if __name__ == "__main__":
    print("\n--- Teste 1: Detecção do Board ---")
    detect_board()

    print("\n--- Teste 2: Detecção Dinâmica da Mão do Hero (mesa_6.png) ---")
    find_and_detect_hero_cards("mesa_6.png")
