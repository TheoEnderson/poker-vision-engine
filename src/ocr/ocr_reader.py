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
    DEFAULT_DEALER_THRESHOLD,
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


def _parse_pot_reading(raw_text: str) -> Tuple[Optional[float], str]:
    """Última linha de valor; nunca extrai um fragmento de um token ilegível."""
    lines = [line.strip().upper() for line in raw_text.splitlines() if line.strip()]
    while lines and lines[-1] in ("POT", "POTE"):
        lines.pop()
    if not lines:
        return None, "EMPTY"
    match = re.fullmatch(r"(?:POTE?\s*)?([0-9]+|[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,3}(?:\.[0-9]{3})+)", lines[-1])
    if match is None:
        return None, "MALFORMED"
    return float(match.group(1).replace(",", "").replace(".", "")), "NUMERIC"


def _recover_pot_conflict(gray: np.ndarray, readings: list) -> Tuple[Optional[float], Dict[str, Any]]:
    """Recupera somente perda de glifos comprovada na linha isolada do valor.

    Não resolve dígitos diferentes com o mesmo comprimento nem por maioria.
    Exige componentes separados, alinhados e estáveis nos dois limiares.
    Recortes nas bordas e geometria incompatível mantêm a captura incerta;
    isso não constitui um classificador universal de dígitos/fontes.
    """
    evidence = dict(status="UNRESOLVED", reason="Linha numérica/glifos sem cobertura verificável.")
    fixed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    rows = np.flatnonzero(np.any(fixed, axis=1))
    bands = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    if len(bands) < 2 or not len(bands[-1]):
        return None, evidence
    top, bottom = int(bands[-1][0]), int(bands[-1][-1]) + 1
    if top == 0 or bottom == gray.shape[0]:
        return None, evidence
    glyphs = []
    for mask in (fixed, otsu):
        stats = cv2.connectedComponentsWithStats(mask[top:bottom])[2][1:]
        if (not len(stats) or np.any(stats[:, cv2.CC_STAT_TOP] != 0)
                or np.any(stats[:, cv2.CC_STAT_HEIGHT] != bottom - top)
                or np.any(stats[:, cv2.CC_STAT_LEFT] == 0)
                or np.any(stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] >= gray.shape[1])):
            return None, evidence
        glyphs.append(stats)
    if len(glyphs[0]) != len(glyphs[1]):
        return None, evidence
    count = len(glyphs[0])
    evidence["glyph_count"] = count
    values = {r["value"] for r in readings if r["value"] is not None}
    compatible = {v for v in values if math.isfinite(v) and len(str(int(v))) == count}
    if len(compatible) != 1:
        evidence["reason"] = "Mais de um valor cobre os glifos, ou nenhum os cobre; conflito mantido."
        return None, evidence

    left = min(int(s[:, cv2.CC_STAT_LEFT].min()) for s in glyphs)
    right = max(int((s[:, cv2.CC_STAT_LEFT] + s[:, cv2.CC_STAT_WIDTH]).max()) for s in glyphs)
    # Margem proporcional à altura do texto, sem alcançar a linha do rótulo.
    pad = min((bottom - top) // 3, (top - int(bands[-2][-1]) - 1) // 2)
    y1, y2 = max(0, top-pad), min(gray.shape[0], bottom+pad)
    x1, x2 = max(0, left-pad), min(gray.shape[1], right+pad)
    evidence["number_roi"] = dict(top=y1, left=x1, width=x2-x1, height=y2-y1)
    isolated = cv2.resize(gray[y1:y2, x1:x2], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(isolated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raw_text = pytesseract.image_to_string(
        binary, config=f"--psm 7 -c tessedit_char_whitelist={OCR_WHITELIST_DIGITS}") or ""
    value, status = _parse_pot_reading(raw_text)
    readings.append(dict(method="isolated_otsu", raw_text=raw_text, value=value, status=status))
    token = raw_text.strip()
    if (re.fullmatch(r"[0-9]+", token) is None or len(token) != count
            or value not in compatible):
        evidence["reason"] = "OCR isolado não confirma o único valor compatível com todos os glifos."
        return None, evidence
    for reading in readings:
        if reading["value"] is not None and reading["value"] != value:
            reading["rejection_reason"] = f"Leitura não cobre os {count} glifos observados na linha do valor."
    evidence.update(status="RESOLVED", reason="OCR da linha isolada confirma cobertura integral dos glifos.")
    return value, evidence


def extract_pot(
    image_path_or_frame: Union[str, Path, np.ndarray] = "mesa_6.png",
    pot_roi: Optional[Dict[str, int]] = None,
    default_value: Optional[float] = 0.0,
    verbose: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Extrai o pote confrontando pré-processamentos da MESMA captura, sem histórico.

    Leituras conflitantes exigem evidência visual adicional na linha do valor.
    Tokens malformados permanecem incertos; não há votação por maioria.
    Uma variante vazia não refuta outra; concordância não garante acerto do OCR.
    Conflitos não resolvidos retornam None mesmo com default_value; --live usa None.
    """
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(status="UNKNOWN", reason="Imagem/ROI indisponível", readings=[])
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
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants = (
        ("native_fixed", cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]),
        ("native_otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ("enlarged_fixed", cv2.threshold(enlarged, 180, 255, cv2.THRESH_BINARY)[1]),
    )
    tess_config = f"--psm 6 -c tessedit_char_whitelist={OCR_WHITELIST_DIGITS}"
    readings = []
    for method, binary in variants:
        raw_text = pytesseract.image_to_string(binary, config=tess_config) or ""
        value, status = _parse_pot_reading(raw_text)
        readings.append(dict(method=method, raw_text=raw_text, value=value, status=status))

    values = {reading["value"] for reading in readings if reading["value"] is not None}
    pot_value = next(iter(values), None)
    if len(values) > 1:
        status, reason = "CONFLICT", "Leituras numéricas conflitantes na mesma captura; pote não confirmado."
    elif any(reading["status"] == "MALFORMED" for reading in readings):
        status, reason = "AMBIGUOUS", "Token do pote incompleto ou malformado; pote não confirmado."
    elif pot_value is None:
        status, reason = "UNKNOWN", "Nenhum valor do pote reconhecido na captura atual."
    elif not math.isfinite(pot_value) or pot_value > MAX_REALISTIC_POT:
        status, reason = "INVALID", "Valor do pote fora da validação de sanidade existente."
    else:
        status, reason = "READ", "Leitura numérica sem divergências nos pré-processamentos disponíveis."

    resolution = None
    if status == "CONFLICT":
        recovered, resolution = _recover_pot_conflict(gray, readings)
        if recovered is not None:
            pot_value = recovered
            if pot_value > MAX_REALISTIC_POT:
                status, reason = "INVALID", "Valor do pote fora da validação de sanidade existente."
            else:
                status, reason = "READ", resolution["reason"]

    if diagnostics is not None:
        diagnostics.update(status=status, reason=reason, readings=readings,
                           roi=dict(top=y1, left=x1, width=x2-x1, height=y2-y1))
        if resolution is not None:
            diagnostics["resolution"] = resolution
    if status != "READ":
        if verbose:
            print(f"[AVISO OCR] {reason} Leituras: {readings}")
        # A ambiguidade não pode virar zero nem reutilizar um pote anterior.
        return None if status in ("CONFLICT", "AMBIGUOUS") else default_value

    if verbose:
        print("=" * 70)
        print("LEITURA ÓPTICA DO POTE (OCR) - Poker Analytics")
        print("=" * 70)
        print(f"Coordenadas da ROI: {{'top': {top}, 'left': {left}, 'width': {width}, 'height': {height}}}")
        print(f"Leituras do Tesseract: {readings}")
        print(f"Valor numérico do Pote consolidado: {pot_value:.1f} fichas")
        print("=" * 70 + "\n")

    return pot_value


def parse_action_button_text(raw_text: str) -> Dict[str, Any]:
    """Interpreta ações explícitas; ausência de valor não significa CHECK."""
    clean = raw_text.strip().upper()
    unknown = {"is_hero_turn": False, "bet_to_call": None, "action_type": "UNKNOWN", "available_actions": []}
    if re.search(r"LEVANTAR|SENTAR|CHECK\s*/\s*FOLD|CHECK\s+ANY|FOLD\s+TO\s+ANY|AUTO", clean):
        return unknown

    labels = {
        "CHECK": "CHECK", "PASSO": "CHECK", "PASS": "CHECK",
        "CALL": "CALL", "PAGO": "CALL", "PAGAR": "CALL",
        "FOLD": "FOLD", "DESISTO": "FOLD", "DESISTIR": "FOLD",
        "RAISE": "RAISE", "AUMENTO": "RAISE", "AUMENTAR": "RAISE",
        "BET": "BET", "APOSTO": "BET", "APOSTAR": "BET",
        "ALL-IN": "ALL-IN", "ALL IN": "ALL-IN", "TUDO": "ALL-IN",
    }
    matches = list(re.finditer(r"\b(?:" + "|".join(labels) + r")\b", clean))
    actions = list(dict.fromkeys(labels[m.group()] for m in matches))
    if not actions or ("CHECK" in actions and "CALL" in actions):
        return unknown

    values_by_label = {}
    for index, match in enumerate(matches):
        lbl = labels[match.group()]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
        # Só aceita número imediatamente associado ao botão atual
        amount = re.fullmatch(r"\s*([$€£]\s*)?(\d+(?:[.,]\d+)*)\s*", clean[match.end():end])
        if amount:
            raw = amount.group(2)
            # Preserva separadores de milhar e permite valores fracionários.
            parts = re.split(r"[.,]", raw)
            if len(parts) > 2 and (len(parts[0]) > 3 or any(len(p) != 3 for p in parts[1:-1])):
                continue
            if len(parts) > 1 and len(parts[-1]) not in (1, 2, 3):
                continue
            if len(parts) > 1 and len(parts[-1]) in (1, 2):
                raw = "".join(parts[:-1]) + "." + parts[-1]
            else:
                raw = "".join(parts)
            value = float(raw)
            if math.isfinite(value) and value > 0:
                values_by_label.setdefault(lbl, []).append(value)

    all_in_nature = None
    if "ALL-IN" in actions:
        if "CHECK" in actions:
            all_in_nature = "BET"
        elif "CALL" in actions:
            all_in_nature = "RAISE"
        elif actions == ["FOLD", "ALL-IN"]:
            all_in_nature = "UNKNOWN"
        else:
            all_in_nature = "UNKNOWN"

    call_values = values_by_label.get("CALL", [])
    all_in_values = values_by_label.get("ALL-IN", [])

    bet = None
    if "CHECK" in actions:
        bet = 0.0
        action_type = "CHECK"
    elif "CALL" in actions:
        action_type = "CALL"
        if len(call_values) == 1:
            bet = call_values[0]
    elif actions == ["FOLD", "ALL-IN"]:
        # O Replay pode exibir o total da street, incluindo contribuição já
        # feita. O texto sozinho não estabelece CALL incremental/pote elegível.
        action_type = "ALL-IN"
    else:
        action_type = "UNKNOWN"

    result = {
        "is_hero_turn": True,
        "bet_to_call": bet,
        "action_type": action_type,
        "available_actions": actions,
    }
    if all_in_nature is not None:
        result["all_in_nature"] = all_in_nature
        result["all_in_displayed_amount"] = all_in_values[0] if len(all_in_values) == 1 else None
        result["all_in_amount_semantics"] = "UNCONFIRMED"
    return result


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
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING", "available_actions": []}

    h, w = img.shape[:2]
    if w < 600 or h < 400:
        if verbose:
            print("[INFO OCR] Imagem fornecida é um recorte pequeno; botões de ação indisponíveis.")
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING", "available_actions": []}

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
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING", "available_actions": []}

    b, g, r = cv2.split(panel)
    r_int, g_int, b_int = r.astype(int), g.astype(int), b.astype(int)
    
    is_red = (r_int - g_int > 50) & (r_int - b_int > 50) & (r > 120)
    is_green = (g_int - r_int > 30) & (g_int - b_int > 30) & (g > 120)
    is_blue = (b_int - r_int > 30) & (b_int - g_int > 30) & (b > 120)
    is_orange = (r_int > 150) & (g_int > 100) & (b_int < 100)
    
    button_mask = (is_red | is_green | is_blue | is_orange).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(button_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    action_buttons = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Relax constraints to account for different browser zooms/resolutions and non-red buttons
        if area > 2000 and bh > 25:
            action_buttons.append((bx, by, bw, bh, area))

    if not action_buttons:
        if verbose:
            print("=" * 70)
            print("DETECÇÃO DE TURNO DO HERO - Poker Analytics")
            print("=" * 70)
            print("Status: Botões de ação não detectados -> Vez do Hero não confirmada (WAITING)")
            print("=" * 70 + "\n")
        return {"is_hero_turn": False, "bet_to_call": 0.0, "action_type": "WAITING", "available_actions": []}

    action_buttons.sort(key=lambda b: b[0])

    # Mantém cada valor ligado ao seu botão, sem misturar CALL com RAISE/slider.
    x_min = min(b[0] for b in action_buttons)
    y_min = min(b[1] for b in action_buttons)
    x_max = max(b[0] + b[2] for b in action_buttons)
    y_max = max(b[1] + b[3] for b in action_buttons)

    gx = search_x1 + x_min
    gy = search_y1 + y_min
    gw = x_max - x_min
    gh = y_max - y_min

    texts = []
    for bx, by, bw, bh, area in action_buttons:
        btn_crop = panel[by:by + bh, bx:bx + bw]
        gray = cv2.cvtColor(btn_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        if parse_action_button_text(text)["is_hero_turn"]:
            texts.append(text.strip())
    raw_text = "\n".join(texts)
    result = parse_action_button_text(raw_text)

    if verbose:
        print("=" * 70)
        print("DETECÇÃO DE TURNO DO HERO E APOSTA (OCR) - Poker Analytics")
        print("=" * 70)
        print(f"Botões de ação ativos detectados: {len(action_buttons)}")
        print(f"Coordenadas do bloco de botões: top={gy}, left={gx}, width={gw}, height={gh}")
        print(f"Texto extraído do bloco: {repr(raw_text.strip())}")
        print(f"Decisão/Status interpretado: {result}")
        print("=" * 70 + "\n")

    return result


def parse_badge_crop(badge_crop: np.ndarray, require_confirmation: bool = False) -> Optional[float]:
    """
    Extrai o valor numérico de fichas da região recortada do badge do Hero.
    Suporta tanto texto padrão (branco/ouro) quanto texto destacado na vez do jogador (verde neon).
    Modo confirmado exige concordância entre dois pré-processamentos; divergência
    ou leitura isolada retorna None. Não escolhe automaticamente o maior valor.
    """
    if badge_crop is None or badge_crop.size == 0:
        return None

    def extract_num(text: str) -> Optional[float]:
        if not text:
            return None
        lines = text.strip().split("\n")
        # Analisa de baixo para cima (fichas ficam na linha inferior do badge)
        for line in reversed(lines):
            if require_confirmation:
                # Dígitos fragmentados ("241 5") não autorizam selecionar apenas "5".
                if not re.fullmatch(r"\s*\d+(?:[.,]\d+)*\s*", line):
                    continue
                parts = re.split(r"[.,]", line.strip())
                if len(parts) > 1:
                    if len(parts) > 2 and (len(parts[0]) > 3 or any(len(p) != 3 for p in parts[1:-1])):
                        continue
                    if len(parts[-1]) in (1, 2):
                        raw = "".join(parts[:-1]) + "." + parts[-1]
                    elif len(parts[-1]) == 3 and len(parts[0]) <= 3:
                        raw = "".join(parts)
                    else:
                        continue
                else:
                    raw = parts[0]
                value = float(raw)
                if math.isfinite(value) and 0 <= value < 50_000_000:
                    return value
                continue
            nums = re.findall(r"\b\d+(?:[.,]\d+)*\b", line)
            if nums:
                raw = nums[-1].replace(",", "").replace(".", "")
                try:
                    val = float(raw)
                    if 0 < val < 50_000_000:
                        return val
                except ValueError:
                    continue
        return None

    def readings():
        # Mesmas rotas visuais existentes; o modo legado mantém o retorno rápido.
        yield extract_num(pytesseract.image_to_string(badge_crop))
        if len(badge_crop.shape) == 3:
            g = cv2.resize(badge_crop[:, :, 1], None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            yield extract_num(pytesseract.image_to_string(g))

        bh, bw = badge_crop.shape[:2]
        bottom = badge_crop[int(bh * 0.38):, :]
        bgray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY) if len(bottom.shape) == 3 else bottom
        up = cv2.resize(bgray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        for th in [50, 60, 75, 90]:
            _, b = cv2.threshold(up, th, 255, cv2.THRESH_BINARY)
            pad = cv2.copyMakeBorder(b, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
            for psm in [6, 7, 11]:
                txt = pytesseract.image_to_string(
                    pad,
                    config=f"--psm {psm} -c tessedit_char_whitelist={OCR_WHITELIST_DIGITS}"
                ).strip()
                yield extract_num(txt)

    candidate = None
    for value in readings():
        if value is None:
            continue
        if not require_confirmation:
            return value
        if candidate is not None:
            return value if value == candidate else None
        candidate = value

    return None


def extract_hero_stack(
    image_path_or_frame: Union[str, Path, np.ndarray],
    last_stack: float = 0.0,
    identifier: str = HERO_IDENTIFIER,
    hero_coords: Optional[Dict[str, int]] = None,
    require_confirmation: bool = False,
) -> Optional[float]:
    """
    Extrai o stack dinâmico do Hero na mesa (valor numérico no badge do jogador).
    Utiliza cache de coordenadas do badge para evitar buscas completas a cada captura.
    Suporta leitura em repouso e durante o turno ativo (destaque verde neon).
    Com require_confirmation=True, falha/divergência retorna None, nunca last_stack.
    """
    global last_hero_name_coords

    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        return None if require_confirmation else last_stack

    h, w = img.shape[:2]
    scale = h / 1080.0

    # 1. Rota Rápida via Cache: se as coordenadas do badge já são conhecidas
    if last_hero_name_coords is not None:
        bx1, by1, bx2, by2 = last_hero_name_coords
        if 0 <= by1 < by2 <= h and 0 <= bx1 < bx2 <= w:
            val = parse_badge_crop(img[by1:by2, bx1:bx2], require_confirmation=require_confirmation)
            if val is not None:
                return val
        # Se falhou, invalida o cache para refazer a busca
        last_hero_name_coords = None

    # 2. Dica de Proximidade das Cartas do Hero (se hero_coords estiver disponível)
    if hero_coords is not None:
        hx = hero_coords.get("left", 0)
        hy = hero_coords.get("top", 0)
        hw = hero_coords.get("width", 100)
        hh = hero_coords.get("height", 80)
        cx1 = max(0, hx - int(50 * scale))
        cx2 = min(w, hx + hw + int(50 * scale))
        cy1 = max(0, hy - int(30 * scale))
        cy2 = min(h, hy + hh + int(90 * scale))
        val = parse_badge_crop(img[cy1:cy2, cx1:cx2], require_confirmation=require_confirmation)
        if val is not None:
            last_hero_name_coords = (cx1, cy1, cx2, cy2)
            return val

    # 3. Busca Completa no Tabuleiro pelo Identificador do Hero
    id_lower = identifier.lower()
    id_clean = re.sub(r"[^a-z0-9]", "", id_lower)
    raw_parts = [re.sub(r"[^a-z0-9]", "", p) for p in re.split(r"[_ \-]+", id_lower)]
    parts = [p for p in raw_parts if len(p) >= 3]
    distinctive = [p for p in parts if len(p) >= 4 and p not in ("the", "and", "for", "end", "poker")]
    if not distinctive:
        distinctive = [p for p in parts if len(p) >= 3]
    if not distinctive and id_clean:
        distinctive = [id_clean]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)

    for i, t in enumerate(data["text"]):
        tx, ty, tw, th = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # Ignora barra superior do navegador e botões de ação na borda inferior
        if ty < 0.08 * h or ty > 0.85 * h:
            continue
        clean = re.sub(r"[^a-zA-Z0-9_]", "", t).lower()
        if "aumento" in clean or "fold" in clean or "check" in clean:
            continue

        is_hero_token = (id_clean and id_clean in clean) or any(p in clean for p in distinctive)
        if is_hero_token:
            bx1 = max(0, tx - int(55 * scale))
            bx2 = min(w, tx + tw + int(60 * scale))
            by1 = max(0, ty - int(15 * scale))
            by2 = min(h, ty + th + int(38 * scale))
            val = parse_badge_crop(img[by1:by2, bx1:bx2], require_confirmation=require_confirmation)
            if val is not None:
                last_hero_name_coords = (bx1, by1, bx2, by2)
                return val

    return None if require_confirmation else last_stack


def extract_hero_position(image_path_or_frame: Union[str, Path, np.ndarray], verbose: bool = False,
                          diagnostics: Optional[Dict[str, Any]] = None,
                          hero_coords: Optional[Dict[str, int]] = None) -> str:
    """
    Identifica a posição do Hero na mesa (BTN, SB, BB, UTG, MP, CO)
    baseado na posição do botão de Dealer (D) e do nome do Hero.
    Assume uma mesa 6-max (layout padrão).
    Valida o disco inteiro e o D vermelho na escala observada. A posição é
    relativa à geometria 6-max; não infere ocupação/folds dos demais assentos.
    """
    if diagnostics is not None:
        diagnostics.update(confidence=None, reason="imagem indisponível", layout="6-max", validated=False)
    img = _resolve_image_input(image_path_or_frame)
    if img is None:
        return "Desconhecida"

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Encontrar o Botão de Dealer
    template_path = PROJECT_ROOT / "assets" / "templates" / "dealer_button.png"
    if not template_path.exists():
        if diagnostics is not None:
            diagnostics["reason"] = "template do dealer ausente"
        if verbose:
            print("[AVISO OCR] Template do botão de Dealer não encontrado.")
        return "Desconhecida"

    template = cv2.imread(str(template_path))
    th, tw = template.shape[:2]
    dealer_angle = 0.0
    dealer_found = False
    best_score = 0.0

    # Em mesas full-screen (w >= 600, h >= 400), foca na área da mesa para evitar falso positivo em barras periféricas
    if w >= 600 and h >= 400:
        y_offset, x_offset = 50, 60
        search_img = img[y_offset:h - 80, x_offset:w - 60]
        # Um match 16x16 dentro de uma carta não é o disco D de 28x28.
        # Localiza componentes brancos inteiros e normaliza sua escala antes
        # de comparar. O D vermelho deve concordar separadamente com o template.
        hsv = cv2.cvtColor(search_img, cv2.COLOR_BGR2HSV)
        white = ((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        def red_glyph(image):
            b, g, r = cv2.split(image.astype(np.int16))
            return ((r - g > 50) & (r - b > 50) & (r > 120)).astype(np.uint8) * 255
        template_red = red_glyph(template)
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if min(cw, ch) < min(tw, th) // 2:
                continue
            gy = y + y_offset
            gx = x + x_offset
            if gy > 0.85 * h or gy < 0.08 * h:
                continue
            if gx < 350 and gy > 0.75 * h:
                continue
            patch = cv2.resize(search_img[y:y+ch, x:x+cw], (tw, th))
            val = float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            glyph_score = float(cv2.matchTemplate(red_glyph(patch), template_red, cv2.TM_CCOEFF_NORMED)[0, 0])
            score = min(val, glyph_score)
            best_score = max(best_score, score)
            if score >= DEFAULT_DEALER_THRESHOLD:
                candidates.append(dict(confidence=val, glyph_score=glyph_score,
                                       box=[gx, gy, cw, ch], center=[gx + cw // 2, gy + ch // 2]))

        if diagnostics is not None:
            diagnostics['candidates'] = candidates
        if len(candidates) > 1:
            if diagnostics is not None:
                diagnostics.update(confidence=best_score, reason="múltiplos discos D; dealer ambíguo")
            return "Desconhecida"
        if len(candidates) == 1:
            selected = candidates[0]
            best_score = selected['confidence']
            dealer_x, dealer_y = selected['center']
            if diagnostics is not None:
                diagnostics.update(dealer_box=selected['box'], dealer_center=selected['center'],
                                   glyph_score=selected['glyph_score'])
            dealer_angle = math.degrees(math.atan2(dealer_y - cy, dealer_x - cx))
            dealer_found = True
    else:
        y_offset, x_offset = 0, 0
        search_img = img
        res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        best_score = float(max_val)
        if max_val >= DEFAULT_DEALER_THRESHOLD:
            dealer_x = max_loc[0] + x_offset + tw // 2
            dealer_y = max_loc[1] + y_offset + th // 2
            dealer_angle = math.degrees(math.atan2(dealer_y - cy, dealer_x - cx))
            dealer_found = True

    if diagnostics is not None:
        diagnostics.update(confidence=best_score,
                           validated=dealer_found,
                           reason="dealer detectado" if dealer_found else "score do dealer abaixo do limiar")

    if not dealer_found:
        if verbose:
            print("[AVISO OCR] Botão de Dealer não detectado na tela.")
        return "Desconhecida"

    # 2. Encontrar o Hero (Nome)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global last_hero_name_coords
    hero_angle = 0.0
    found_hero = False

    if hero_coords is not None:
        hx = hero_coords.get("left", 0)
        hy = hero_coords.get("top", 0)
        hw = hero_coords.get("width", 0)
        hh = hero_coords.get("height", 0)
        if hw > 0 and hh > 0:
            center_hx = hx + hw // 2
            center_hy = hy + hh // 2
            hero_angle = math.degrees(math.atan2(center_hy - cy, center_hx - cx))
            found_hero = True

    if not found_hero and last_hero_name_coords is not None:
        bx1, by1, bx2, by2 = last_hero_name_coords
        center_hx, center_hy = (bx1 + bx2) // 2, (by1 + by2) // 2
        hero_angle = math.degrees(math.atan2(center_hy - cy, center_hx - cx))
        found_hero = True
    elif not found_hero:
        id_lower = HERO_IDENTIFIER.lower()
        id_clean = re.sub(r"[^a-z0-9]", "", id_lower)
        raw_parts = [re.sub(r"[^a-z0-9]", "", p) for p in re.split(r"[_ \-]+", id_lower)]
        parts = [p for p in raw_parts if len(p) >= 3]
        distinctive = [p for p in parts if len(p) >= 4 and p not in ("the", "and", "for", "end", "poker")]
        if not distinctive:
            distinctive = [p for p in parts if len(p) >= 3]
        if not distinctive and id_clean:
            distinctive = [id_clean]

        data = pytesseract.image_to_data(gray, output_type=Output.DICT)
        for i, text in enumerate(data["text"]):
            clean = re.sub(r"[^a-zA-Z0-9_]", "", text).lower()
            if "aumento" in clean or "fold" in clean or "check" in clean:
                continue
            if (id_clean and id_clean in clean) or any(p in clean for p in distinctive):
                center_hx = data["left"][i] + data["width"][i] // 2
                center_hy = data["top"][i] + data["height"][i] // 2
                hero_angle = math.degrees(math.atan2(center_hy - cy, center_hx - cx))
                found_hero = True
                break

    if not found_hero:
        if diagnostics is not None:
            diagnostics["reason"] = "dealer encontrado, mas nome/assento do Hero indisponível"
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
    if diagnostics is not None:
        diagnostics.update(reason="posição inferida no layout 6-max", position=position,
                           dealer_seat=ideal_seats_6[dealer_idx][1], hero_seat=ideal_seats_6[hero_idx][1],
                           dealer_angle=dealer_angle, hero_angle=hero_angle,
                           position_basis="geometria 6-max; ocupação dos assentos não confirmada")

    if verbose:
        print(f"[OCR] Dealer={dealer_angle:.1f}° (idx {dealer_idx}), Hero={hero_angle:.1f}° (idx {hero_idx}) -> Posição: {position}")

    return position


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
