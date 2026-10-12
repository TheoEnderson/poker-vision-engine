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
from src.diagnostics import timed

from src.config import (
    CARD_BRIGHTNESS_THRESHOLD,
    DEFAULT_NUM_OPPONENTS,
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
    board_coords: Optional[Dict[str, int]] = None,
    save_unknown: bool = True,
) -> List[str]:
    """Detecta dinamicamente as cartas do board comunitário na imagem da mesa usando contornos."""
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

    # Recorta uma região ampla que engloba o board independentemente de variações de scroll
    if width > 600 and height > 400:
        bt = max(0, 350)
        bl = max(0, 450)
        bh = min(height - bt, 350)
        bw = min(width - bl, 1000)
        board_roi = board[bt : bt + bh, bl : bl + bw]
    else:
        board_roi = board

    gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(board_roi, cv2.COLOR_BGR2HSV)
    white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
    white_u8 = (white_mask.astype(np.uint8)) * 255

    cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        cbx, cby, cbw, cbh = cv2.boundingRect(c)
        if 2000 < area < 15000 and 40 < cbw < 150 and 50 < cbh < 150:
            # Elimina componentes muito curtos e largos ou altos e finos que não parecem cartas
            ratio = cbh / float(cbw)
            if 0.8 < ratio < 1.8:
                candidates.append((cbx, cby, cbw, cbh))

    candidates.sort(key=lambda b: b[0])
    detected_board = []

    for i, (cbx, cby, cbw, cbh) in enumerate(candidates):
        if i >= num_slots:
            break

        slot_img = board_roi[cby : cby + cbh, cbx : cbx + cbw]
        
        corner_h = min(cbh, max(60, int(round(cbh * 0.55))))
        corner_w = min(cbw, int(round(cbw * 0.45)))
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
            if save_unknown:
                save_unknown_card(slot_img, prefix=f"board_slot_{i + 1}", verbose=verbose)
            if verbose:
                print(
                    f"Slot {i + 1}: Confiança insuficiente (Rank: {best_rank_score:.3f}, "
                    f"Naipe: {best_suit_score:.3f}) -> {card_str}"
                )

        detected_board.append(card_str)

    while len(detected_board) < num_slots:
        detected_board.append("Vazio")

    if verbose:
        print("-" * 70)
        print(f"Board detectado: {detected_board}")
        print("=" * 70 + "\\n")

    return detected_board


def _hero_suit_region(glyph_mask: np.ndarray, red_suit: bool = True) -> Optional[np.ndarray]:
    """Localiza o naipe abaixo do rank, sem truncar pela altura visível da carta.

    O painel do jogador pode cobrir o fundo da carta. A altura visível não é,
    portanto, uma referência segura para o fim do símbolo do naipe.
    """
    height, width = glyph_mask.shape
    corner = glyph_mask[:, :max(1, int(width * 0.4))].astype(np.uint8) * 255
    contours, _ = cv2.findContours(corner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Descarta pedaços dos desenhos centrais que atravessam a coluna.
        if cv2.contourArea(contour) > 5 and x > 0 and x + w < corner.shape[1]:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: (box[1], box[0]))
    if not boxes:
        return None

    # Um rank como '10' pode ter dois componentes na mesma faixa vertical.
    rank_bottom = boxes[0][1] + boxes[0][3]
    rank_boxes = [box for box in boxes if box[1] < rank_bottom]
    rank_bottom = max(y + h for x, y, w, h in rank_boxes)
    if not red_suit:
        # Em cartas pretas, a antialiasing do naipe pode unir seu componente à
        # borda cinza inferior. Busca o template abaixo do rank sem exigir que
        # o contorno esteja isolado da borda (nem cortar sua base em 75%).
        region = corner[rank_bottom + 1:, :]
        if not region.size:
            return None
        return cv2.copyMakeBorder(region, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
    suit_boxes = [box for box in boxes if box[1] >= rank_bottom]
    if not suit_boxes:
        return None
    x, y, w, h = suit_boxes[0]
    # Copas/ouros podem terminar exatamente no recorte. Aceita a ponta estreita,
    # mas mantém incerto um corte largo pelo corpo do símbolo.
    if y + h >= height:
        bottom_width = np.count_nonzero(corner[-1, x:x + w])
        if bottom_width > max(2, w * 0.3):
            return None
    return cv2.copyMakeBorder(corner[y:y + h, x:x + w], 2, 2, 2, 2, cv2.BORDER_CONSTANT)


def find_and_detect_hero_cards(
    image_path_or_frame: Union[str, Path, np.ndarray] = "mesa_6.png",
    board_coords: Optional[Dict[str, int]] = None,
    threshold: float = DEFAULT_RANK_THRESHOLD,
    verbose: bool = True,
    hero_coords_hint: Optional[Dict[str, int]] = None,
    return_coords: bool = False,
    save_unknown: bool = True,
    diagnostics: Optional[Dict] = None,
) -> Union[List[str], Tuple[List[str], Optional[Dict[str, int]]]]:
    """
    Localiza dinamicamente as duas cartas abertas do Hero em qualquer posição da mesa
    (6-max ou 9-max), ignorando as cartas comunitárias do board.
    Se hero_coords_hint for fornecido, foca no assento indicado sem tentar outro
    jogador quando a leitura falha. Sem assento, múltiplas mãos são ambíguas.
    """
    if board_coords is None:
        board_coords = {"top": 415, "left": 734, "width": 449, "height": 109}
    if diagnostics is None:
        diagnostics = {}
    diagnostics.clear()
    diagnostics.update(contours=[], slots=[], rejection=None)

    ranks, suits = load_templates()

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        if verbose:
            print(f"[ERRO] Não foi possível carregar a imagem de: {image_path_or_frame}")
        return ([], None) if return_coords else []

    h, w = img.shape[:2]
    if w < 600 or h < 400:
        if verbose:
            print("[INFO] Imagem fornecida é um recorte do board, não uma mesa completa com assentos.")
        return ([], None) if return_coords else []

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
        if area > 1500:
            diagnostics["contours"].append(dict(box=[cbx, cby, cbw, cbh], area=area))
        if area > 1500 and 40 < cbw < 250 and cbh >= 120:
            # Nas capturas reais, a borda estreita do avatar une-se à carta.
            # A abertura 5x5 separa essa borda (3x3 não basta) SOMENTE da
            # máscara de localização; os pixels usados na classificação são
            # os originais. Mantém os limites geométricos das cartas.
            component = white_u8[cby:cby + cbh, cbx:cbx + cbw]
            separated = cv2.morphologyEx(component, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            parts, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for part in parts:
                px, py, pw, ph = cv2.boundingRect(part)
                if 1500 < cv2.contourArea(part) < 10000 and 40 < pw < 120 and 40 < ph < 120:
                    candidates.append((cbx + px, cby + py, pw, ph))
                    diagnostics.setdefault("separated_cards", []).append([cbx + px, cby + py, pw, ph])
        if 1500 < area < 10000 and 40 < cbw < 120 and 40 < cbh < 120:
            candidates.append((cbx, cby, cbw, cbh))
        elif 3000 < area < 20000 and 120 < cbw < 250 and 40 < cbh < 120:
            candidates.append((cbx, cby, cbw, cbh))

    @timed("hero_classification")
    def match_hero_slot(slot_img: np.ndarray, is_second_slot: bool = False):
        sgray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
        shsv = cv2.cvtColor(slot_img, cv2.COLOR_BGR2HSV)

        black_mask = (sgray < 185)
        h, s, v = shsv[:, :, 0], shsv[:, :, 1], shsv[:, :, 2]
        red_mask = ((h < 15) | (h > 165)) & (s > 90) & (v > 80)
        sbin = ((black_mask | red_mask).astype(np.uint8)) * 255
        rank_cutoff = max(36, int(slot_img.shape[0] * 0.45))
        best_r, best_r_score = match_glyph_multiscale(sbin[:rank_cutoff, :], ranks)

        corner = slot_img[:, :max(1, int(slot_img.shape[1] * 0.4))]
        candidate_suits = filter_suits_by_color(corner, suits)
        red_suit = set(candidate_suits).issubset({"h", "d"})
        glyph_mask = red_mask if red_suit else black_mask
        suit_reg = _hero_suit_region(glyph_mask, red_suit=red_suit)
        # Não compara um pedaço de coração com o template inteiro de ouros.
        if suit_reg is None:
            best_s, best_s_score = None, -1.0
        else:
            best_s, best_s_score = match_glyph_multiscale(suit_reg, candidate_suits)

        local_thresh = threshold
        card_str = (
            f"{best_r}{best_s}"
            if (best_r_score >= local_thresh and best_s_score >= local_thresh)
            else "Desconhecida"
        )
        diagnostics["slots"].append(dict(card=card_str, rank=best_r, rank_score=float(best_r_score),
                                         suit=best_s, suit_score=float(best_s_score),
                                         suit_region_found=suit_reg is not None))
        return card_str, best_r, best_r_score, best_s, best_s_score

    candidate_boxes = []
    # 1. Candidatos unidos (par de cartas junto)
    for c in candidates:
        if 110 <= c[2] <= 260 and 40 <= c[3] <= 120:
            candidate_boxes.append(c)

    # 2. Candidatos separados (duas cartas lado a lado)
    if len(candidates) >= 2:
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c1, c2 = sorted([candidates[i], candidates[j]], key=lambda b: b[0])
                if abs(c1[1] - c2[1]) < 35 and abs((c1[0] + c1[2]) - c2[0]) < 55:
                    hx = c1[0]
                    hy = min(c1[1], c2[1])
                    hw = (c2[0] + c2[2]) - hx
                    hh = max(c1[1] + c1[3], c2[1] + c2[3]) - hy
                    candidate_boxes.append((hx, hy, hw, hh))

    if hero_coords_hint:
        hl = hero_coords_hint.get("left", 0)
        ht = hero_coords_hint.get("top", 0)
        # Prioriza e filtra candidatos próximos ao assento do Hero (< 250px),
        # impedindo que cartas de oponentes expostas no Showdown sejam confundidas com as do Hero
        close_boxes = [b for b in candidate_boxes if ((b[0] - hl) ** 2 + (b[1] - ht) ** 2) <= (250 ** 2)]
        if hero_coords_hint.get("width", 0) > 0 and hero_coords_hint.get("height", 0) > 0:
            hint_right = hl + hero_coords_hint["width"]
            hint_bottom = ht + hero_coords_hint["height"]
            # Uma caixa conhecida é evidência mais forte que a tolerância de 250px.
            close_boxes = [b for b in close_boxes if (
                max(0, min(hint_right, b[0] + b[2]) - max(hl, b[0]))
                * max(0, min(hint_bottom, b[1] + b[3]) - max(ht, b[1]))
            ) >= 0.5 * min(
                hero_coords_hint["width"] * hero_coords_hint["height"], b[2] * b[3]
            )]
        if close_boxes:
            close_boxes.sort(key=lambda b: (b[0] - hl) ** 2 + (b[1] - ht) ** 2)
            # O assento mais próximo continua sendo o alvo mesmo se estiver ilegível.
            candidate_boxes = close_boxes[:1]
        else:
            diagnostics["rejection"] = "NO_CANDIDATE_AT_KNOWN_SEAT"
            if verbose:
                print("[INFO] Nenhuma carta aberta detectada no assento conhecido do Hero.")
            return ([], None) if return_coords else []

    if not candidate_boxes:
        diagnostics["rejection"] = "NO_CARD_PAIR_GEOMETRY"
        if verbose:
            print("[INFO] Nenhuma carta aberta do Hero detectada fora do board (Fold ou mesa vazia).")
        return ([], None) if return_coords else []

    best_cards = []
    best_box = None
    best_res = None
    last_res = None

    for box in candidate_boxes:
        hx, hy, hw, hh = box
        hero_crop = img[hy : hy + hh, hx : hx + hw]
        half_w = hw // 2
        slot1 = hero_crop[:, :half_w]
        slot2 = hero_crop[:, half_w:]

        c1_str, r1, sr1, s1, ss1 = match_hero_slot(slot1, is_second_slot=False)
        c2_str, r2, sr2, s2, ss2 = match_hero_slot(slot2, is_second_slot=True)
        last_res = (box, slot1, slot2, c1_str, r1, sr1, s1, ss1, c2_str, r2, sr2, s2, ss2)

        if c1_str != "Desconhecida" and c2_str != "Desconhecida":
            if best_box is not None:
                # Caixas duplicadas/sobrepostas podem representar o mesmo par.
                bx, by, bw, bh = best_box
                overlap = max(0, min(bx + bw, hx + hw) - max(bx, hx)) * max(
                    0, min(by + bh, hy + hh) - max(by, hy)
                )
                if overlap < 0.5 * min(bw * bh, hw * hh):
                    diagnostics["rejection"] = "MULTIPLE_SEATS_WITH_OPEN_CARDS"
                    if verbose:
                        print("[INFO] Múltiplas mãos abertas sem assento do Hero definido. Retornando [].")
                    return ([], None) if return_coords else []
                continue
            best_cards = [c1_str, c2_str]
            best_box = box
            best_res = last_res

    if not best_cards:
        diagnostics["rejection"] = "CLASSIFICATION_UNCERTAIN"
        if last_res:
            box, slot1, slot2, c1_str, r1, sr1, s1, ss1, c2_str, r2, sr2, s2, ss2 = last_res
            if save_unknown and c1_str == "Desconhecida":
                save_unknown_card(slot1, prefix="hero_slot_1", verbose=verbose)
            if save_unknown and c2_str == "Desconhecida":
                save_unknown_card(slot2, prefix="hero_slot_2", verbose=verbose)
            if verbose:
                print(f"[INFO] Cartas do Hero não reconhecidas com confiança suficiente: [{c1_str}, {c2_str}]. Retornando [].")
        return ([], None) if return_coords else []

    hero_coords_dict = {"top": int(best_box[1]), "left": int(best_box[0]), "width": int(best_box[2]), "height": int(best_box[3])}
    hero_cards = best_cards

    if verbose:
        box, slot1, slot2, c1_str, r1, sr1, s1, ss1, c2_str, r2, sr2, s2, ss2 = best_res
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

    if return_coords:
        return hero_cards, hero_coords_dict
    return hero_cards


def detect_active_opponents(
    image_path_or_frame: Union[str, Path, np.ndarray],
    verbose: bool = False,
    hero_coords: Optional[Dict[str, int]] = None,
) -> int:
    """
    Estima adversários ainda na mão, excluindo o Hero, pelos pares de dorsos
    vermelhos visíveis. Não conta nomes/avatares nem todos os assentos ocupados.
    Suporta 1 a 5 adversários no layout atual. Sem evidência suficiente, retorna
    DEFAULT_NUM_OPPONENTS como estimativa, não como confirmação de jogadores.
    """
    img = _resolve_image_input(image_path_or_frame)
    if img is None or img.size == 0:
        return DEFAULT_NUM_OPPONENTS

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        # Dorsos são regiões retangulares quase preenchidas de vermelho.
        # Naipes e partes de avatares não são evidência de um jogador ativo.
        if not (150 < area < 10000 and 15 < w < 120 and 40 < h < 120 and area / (w * h) > 0.85):
            continue
        gx, gy = x + w // 2, y + h // 2
        if gy > 800:
            continue  # Botões de ação
        if 350 < gy < 600 and 600 < gx < 1300:
            continue  # Board
        if hero_coords and (
            hero_coords["left"] <= gx < hero_coords["left"] + hero_coords["width"]
            and hero_coords["top"] <= gy < hero_coords["top"] + hero_coords["height"]
        ):
            continue
        boxes.append((x, y, w, h))

    def adjacent(a, b):
        left, right = sorted((a, b), key=lambda box: box[0])
        gap = right[0] - (left[0] + left[2])
        return (
            0 <= gap <= 25
            and abs(left[1] - right[1]) <= 0.25 * min(left[3], right[3])
            and min(left[3], right[3]) >= 0.75 * max(left[3], right[3])
        )

    # Agrupa pelo espaço entre as bordas, não pela distância entre os centros:
    # duas cartas inteiramente visíveis também pertencem ao mesmo jogador.
    valid_opponents = 0
    pending = list(boxes)
    while pending:
        group = [pending.pop()]
        for anchor in group:
            neighbours = [box for box in pending if adjacent(anchor, box)]
            for box in neighbours:
                pending.remove(box)
            group.extend(neighbours)
        # Um único retângulo (ex.: baralho decorativo no centro) não é um par.
        if len(group) >= 2:
            valid_opponents += 1

    if valid_opponents == 0:
        if verbose:
            print(f"[INFO] Contagem de adversários indisponível; usando estimativa {DEFAULT_NUM_OPPONENTS}.")
        return DEFAULT_NUM_OPPONENTS

    # Numa mesa 6-max, se tirarmos o Hero, no máximo 5 oponentes
    return min(valid_opponents, 5)


if __name__ == "__main__":
    print("\n--- Teste 1: Detecção do Board ---")
    detect_board()

    print("\n--- Teste 2: Detecção Dinâmica da Mão do Hero (mesa_6.png) ---")
    find_and_detect_hero_cards("mesa_6.png")
