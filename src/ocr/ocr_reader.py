"""
src/ocr/ocr_reader.py
Módulo de Reconhecimento Óptico de Caracteres (OCR) para extração de Pote, Stack do Hero e Ações da Mesa.
"""

import os
import re
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from src.config import (
    ACTION_BUTTONS_ROI,
    HERO_IDENTIFIER,
    MAX_REALISTIC_POT,
    OCR_WHITELIST_DIGITS,
    POT_ROI,
    PROJECT_ROOT,
    SAMPLES_DIR,
)

# Estado global de cache para ROI do nome do Hero
last_hero_name_coords: Optional[Tuple[int, int, int, int]] = None


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


def extract_pot(
    image_path_or_frame: Union[str, Path, np.ndarray] = "mesa_6.png",
    pot_roi: Optional[Dict[str, int]] = None,
    default_value: float = 0.0,
    verbose: bool = True
) -> float:
    """
    Extrai numericamente o valor do pote atual da mesa utilizando Visão Computacional e OCR.
    """
    if pot_roi is None:
        pot_roi = dict(POT_ROI)

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        if verbose:
            print(f"[ERRO OCR] Não foi possível carregar imagem de: {image_path_or_frame}")
        return default_value

    h, w = img.shape[:2]
    if w < 600 or h < 400:
        if verbose:
            print("[INFO OCR] Imagem fornecida é um recorte pequeno; área do pote indisponível.")
        return default_value

    top = pot_roi.get("top", 210)
    left = pot_roi.get("left", 850)
    width = pot_roi.get("width", 220)
    height = pot_roi.get("height", 60)

    effective_height = max(height, 95) if (top <= 230 and height <= 75) else height

    y1 = max(0, min(top, h - 1))
    y2 = max(0, min(top + effective_height, h))
    x1 = max(0, min(left, w - 1))
    x2 = max(0, min(left + width, w))

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        if verbose:
            print("[ERRO OCR] Recorte da ROI do pote resultou em imagem vazia.")
        return default_value

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    tess_config = f"--psm 6 -c tessedit_char_whitelist={OCR_WHITELIST_DIGITS}"
    raw_text = pytesseract.image_to_string(thresh, config=tess_config)

    if not raw_text.strip():
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw_text = pytesseract.image_to_string(otsu_thresh, config=tess_config)

    clean_text = raw_text.strip().replace(",", "").replace(".", "")

    if not clean_text:
        if verbose:
            print("[AVISO OCR] Nenhum dígito legível identificado na região do pote. Retornando padrão seguro.")
        return default_value

    try:
        pot_value = float(clean_text)
    except ValueError:
        digits_match = re.search(r"\d+", clean_text)
        if digits_match:
            pot_value = float(digits_match.group(0))
        else:
            pot_value = default_value

    if pot_value > MAX_REALISTIC_POT:
        if verbose:
            print(f"[AVISO OCR] Leitura do Pote exorbitante ({pot_value}). Retornando valor seguro anterior.")
        return default_value

    if verbose:
        print("=" * 70)
        print("LEITURA ÓPTICA DO POTE (OCR) - Poker Analytics")
        print("=" * 70)
        print(f"Coordenadas da ROI: {{'top': {top}, 'left': {left}, 'width': {width}, 'height': {height}}}")
        print(f"Texto detectado pelo Tesseract: {repr(raw_text.strip())}")
        print(f"Valor numérico do Pote consolidado: {pot_value:.1f} fichas")
        print("=" * 70 + "\n")

    return pot_value


def parse_action_button_text(raw_text: str) -> Dict[str, Any]:
    """
    Interpreta o texto extraído via OCR do botão central de ação do Replay Poker.
    """
    clean = raw_text.strip().upper()
    
    if "LEVANTAR" in clean or "SENTAR" in clean:
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING"}
        
    numbers = re.findall(r"\d+(?:[.,]\d+)*", clean)
    if numbers:
        raw_num = numbers[-1].replace(",", "").replace(".", "")
        try:
            bet_val = float(raw_num)
        except ValueError:
            bet_val = 0.0
        return {"is_hero_turn": True, "bet_to_call": bet_val, "action_type": "CALL"}
    elif "PASSO" in clean or "CHECK" in clean or "PASS" in clean:
        return {"is_hero_turn": True, "bet_to_call": 0.0, "action_type": "CHECK"}
    else:
        return {"is_hero_turn": True, "bet_to_call": 0.0, "action_type": "CHECK"}


def detect_turn_and_bet_to_call(
    image_path_or_frame: Union[str, Path, np.ndarray] = "mesa_turno.png",
    action_roi: Optional[Dict[str, int]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analisa a região dos botões de ação e determina se é o turno do Hero e a aposta a pagar.
    """
    if action_roi is None:
        action_roi = dict(ACTION_BUTTONS_ROI)

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        if verbose:
            print(f"[ERRO OCR] Não foi possível carregar imagem de: {image_path_or_frame}")
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING"}

    h, w = img.shape[:2]
    if w < 600 or h < 400:
        if verbose:
            print("[INFO OCR] Imagem fornecida é um recorte pequeno; botões de ação indisponíveis.")
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING"}

    top = action_roi.get("top", 870)
    left = action_roi.get("left", 1200)
    width = action_roi.get("width", 650)
    height = action_roi.get("height", 150)

    if top <= 950:
        search_y1 = max(0, top)
        search_y2 = min(h, max(top + height, 1075))
    else:
        search_y1 = max(0, top)
        search_y2 = min(h, top + height)

    search_x1 = max(0, left)
    search_x2 = min(w, left + width)

    panel = img[search_y1:search_y2, search_x1:search_x2]
    if panel.size == 0:
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING"}

    b, g, r = cv2.split(panel)
    red_mask = (
        (r.astype(int) - g.astype(int) > 75)
        & (r.astype(int) - b.astype(int) > 75)
        & (r > 150)
    ).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_buttons = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Relax constraints to account for different browser zooms/resolutions
        if area > 4000 and bh > 25 and (search_y1 + by >= 800):
            action_buttons.append((bx, by, bw, bh, area))

    if not action_buttons:
        if verbose:
            print("=" * 70)
            print("DETECÇÃO DE TURNO DO HERO - Poker Analytics")
            print("=" * 70)
            print("Status: Botões vermelhos ausentes na mesa -> Turno de outro jogador (WAITING)")
            print("=" * 70 + "\n")
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING"}

    action_buttons.sort(key=lambda b: b[0])

    if len(action_buttons) >= 3:
        mid_btn = action_buttons[1]
    elif len(action_buttons) == 2:
        mid_btn = action_buttons[1]
    else:
        mid_btn = action_buttons[0]

    mbx, mby, mbw, mbh, _ = mid_btn
    gx = search_x1 + mbx
    gy = search_y1 + mby

    btn_crop = img[gy : gy + mbh, gx : gx + mbw]
    gray = cv2.cvtColor(btn_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    raw_text = pytesseract.image_to_string(thresh, config="--psm 6")
    result = parse_action_button_text(raw_text)

    if verbose:
        print("=" * 70)
        print("DETECÇÃO DE TURNO DO HERO E APOSTA (OCR) - Poker Analytics")
        print("=" * 70)
        print(f"Botões de ação ativos detectados: {len(action_buttons)}")
        print(f"Coordenadas do Botão Central: top={gy}, left={gx}, width={mbw}, height={mbh}")
        print(f"Texto extraído do botão central: {repr(raw_text.strip())}")
        print(f"Decisão/Status interpretado: {result}")
        print("=" * 70 + "\n")

    return result


def extract_hero_stack(
    image_path_or_frame: Union[str, Path, np.ndarray],
    last_stack: float = 0.0,
    identifier: str = HERO_IDENTIFIER
) -> float:
    """
    Extrai o stack dinâmico do Hero na mesa (valor numérico abaixo do nome do jogador).
    Utiliza otimização de cache de coordenadas (ROI) para evitar buscas full-frame contínuas.
    """
    global last_hero_name_coords

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        return last_stack

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def process_roi(roi_img: np.ndarray) -> str:
        roi_resized = cv2.resize(roi_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(roi_resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(255 - otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(otsu)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h > 15 and w < 50:
                cv2.rectangle(mask, (max(0, x - 2), max(0, y - 2)), (x + w + 2, y + h + 2), 255, -1)
        otsu = cv2.bitwise_or(otsu, 255 - mask)
        cv2.imwrite("debug_otsu_real.png", otsu)
        return pytesseract.image_to_string(otsu, config=f"--psm 7 -c tessedit_char_whitelist={OCR_WHITELIST_DIGITS}")

    # 1. Tenta usar o cache da última coordenada conhecida
    if last_hero_name_coords is not None:
        hx, hy, hw, hh = last_hero_name_coords
        name_roi = gray[max(0, hy - 10) : hy + hh + 10, max(0, hx - 10) : hx + hw + 10]
        text = pytesseract.image_to_string(name_roi, config="--psm 7")
        if any(part in text for part in ["Ment", "End", "The", identifier]):
            stack_roi = gray[hy + hh : hy + hh + 25, max(0, hx - 150) : min(gray.shape[1], hx + hw + 150)]
            if stack_roi.size > 0:
                text_stack = process_roi(stack_roi)
            else:
                text_stack = "" 
            numbers = re.findall(r"\d+(?:[.,]\d+)*", text_stack.strip())
            if numbers:
                raw = numbers[-1].replace(",", "").replace(".", "")
                try:
                    val = float(raw)
                    if last_stack > 0 and (val > 3 * last_stack or val < 0.3 * last_stack):
                        return last_stack
                    return val
                except ValueError:
                    pass

    # 2. Fallback: Busca na tela inteira
    # Aumentar a resolução ajuda a ler nomes em resoluções menores
    gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    data = pytesseract.image_to_data(otsu, output_type=Output.DICT)
    name_idx = -1
    for i, text in enumerate(data["text"]):
        if any(part in text for part in ["Ment", "End", "The", identifier]):
            name_idx = i
            break

    if name_idx == -1:
        return last_stack

    # Coordenadas da imagem dobrada
    left_res = data["left"][name_idx]
    top_res = data["top"][name_idx]
    width_res = data["width"][name_idx]
    height_res = data["height"][name_idx]
    
    # Reverte para as originais
    left, top, width, height = left_res // 2, top_res // 2, width_res // 2, height_res // 2
    last_hero_name_coords = (left, top, width, height)

    stack_roi = gray[top + height : top + height + 25, max(0, left - 150) : min(gray.shape[1], left + width + 150)]
    if stack_roi.size == 0:
        return last_stack
    text_stack = process_roi(stack_roi)

    clean = text_stack.strip()
    numbers = re.findall(r"\d+(?:[.,]\d+)*", clean)
    if numbers:
        raw_num = numbers[-1].replace(",", "").replace(".", "")
        try:
            val = float(raw_num)
            if last_stack > 0 and (val > 3 * last_stack or val < 0.3 * last_stack):
                return last_stack
            return val
        except ValueError:
            pass

    return last_stack


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTE 1: Extração do Pote via OCR (mesa_6.png)")
    print("=" * 70)
    extracted = extract_pot("mesa_6.png", verbose=True)
    expected = 28.0
    print(f"Resultado Pote: Extraído = {extracted} | Esperado = {expected}")
    assert extracted == expected, f"Erro: esperado {expected}, obteve {extracted}"
    print("[SUCESSO] Valor do pote extraído com 100% de exatidão!\n")

    print("\n" + "=" * 70)
    print("TESTE 2: Detecção do Turno do Hero e Bet to Call (mesa_turno.png)")
    print("=" * 70)
    turn_info_turno = detect_turn_and_bet_to_call("mesa_turno.png", verbose=True)
    print(f"Resultado Turno (mesa_turno.png): {turn_info_turno}")
    assert turn_info_turno["is_hero_turn"] is True
    assert turn_info_turno["bet_to_call"] == 0.0
    assert turn_info_turno["action_type"] == "CHECK"
    print("[SUCESSO] Turno e ação CHECK detectados perfeitamente em mesa_turno.png!\n")

    print("\n" + "=" * 70)
    print("TESTE 3: Verificação de Turno em Espera (mesa_6.png)")
    print("=" * 70)
    turn_info_6 = detect_turn_and_bet_to_call("mesa_6.png", verbose=True)
    print(f"Resultado Turno (mesa_6.png): {turn_info_6}")
    assert turn_info_6["is_hero_turn"] is False
    assert turn_info_6["action_type"] == "WAITING"
    print("[SUCESSO] Estado WAITING detectado corretamente em mesa_6.png!\n")

def extract_hero_position(image_path_or_frame: Union[str, Path, np.ndarray], verbose: bool = False) -> str:
    """
    Identifica a posição do Hero na mesa (BTN, SB, BB, UTG, MP, CO)
    baseado na posição do botão de Dealer (D) e do nome do Hero.
    Assume uma mesa 6-max (layout padrão).
    """
    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        return "Desconhecida"

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Encontrar o Botão de Dealer
    template_path = PROJECT_ROOT / "assets" / "templates" / "dealer_button.png"
    if not template_path.exists():
        if verbose:
            print("[AVISO OCR] Template do botão de Dealer não encontrado.")
        return "Desconhecida"

    template = cv2.imread(str(template_path))
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    dealer_angle = 0.0
    if max_val > 0.6:
        dealer_x = max_loc[0] + template.shape[1] // 2
        dealer_y = max_loc[1] + template.shape[0] // 2
        dealer_angle = math.degrees(math.atan2(dealer_y - cy, dealer_x - cx))
    else:
        if verbose:
            print("[AVISO OCR] Botão de Dealer não detectado na tela.")
        return "Desconhecida"

    # 2. Encontrar o Hero (Nome)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Utilizar a coordenada do cache se existir
    global last_hero_name_coords
    hero_angle = 0.0
    found_hero = False
    
    if last_hero_name_coords is not None:
        hx, hy, hw, hh = last_hero_name_coords
        center_hx, center_hy = hx + hw // 2, hy + hh // 2
        hero_angle = math.degrees(math.atan2(center_hy - cy, center_hx - cx))
        found_hero = True
    else:
        # Busca completa (fallback)
        gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(otsu, output_type=Output.DICT)
        
        for i, text in enumerate(data["text"]):
            if "Ment" in text or "End" in text or HERO_IDENTIFIER in text:
                hx = data["left"][i] // 2 + data["width"][i] // 4
                hy = data["top"][i] // 2 + data["height"][i] // 4
                hero_angle = math.degrees(math.atan2(hy - cy, hx - cx))
                found_hero = True
                break

    if not found_hero:
        if verbose:
            print("[AVISO OCR] Avatar/Nome do Hero não detectado para cálculo de posição.")
        return "Desconhecida"

    # 3. Mapear ângulos para a geometria da mesa 6-max
    ideal_seats_6 = [
        (0, "Mid Right"),
        (45, "Bot Right"),
        (135, "Bot Left"),
        (180, "Mid Left"),
        (-135, "Top Left"),
        (-45, "Top Right")
    ]

    def snap_to_seat(angle, seats):
        angle = (angle + 180) % 360 - 180
        best_seat = 0
        min_diff = 999
        for i, (ideal_angle, name) in enumerate(seats):
            diff = abs((angle - ideal_angle + 180) % 360 - 180)
            if diff < min_diff:
                min_diff = diff
                best_seat = i
        return best_seat

    dealer_idx = snap_to_seat(dealer_angle, ideal_seats_6)
    hero_idx = snap_to_seat(hero_angle, ideal_seats_6)

    positions_6max = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    dist = (hero_idx - dealer_idx) % 6
    position = positions_6max[dist]
    
    if verbose:
        print(f"[OCR] Dealer={dealer_angle:.1f}° (idx {dealer_idx}), Hero={hero_angle:.1f}° (idx {hero_idx}) -> Posição: {position}")

    return position
