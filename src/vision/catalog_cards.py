#!/usr/bin/env python3
"""
catalog_cards.py - Utilitário para catalogação de cartas desconhecidas do Poker Analytics.

Funcionalidades:
1. Agrupa imagens duplicadas em assets/unknown_cards/ por hash MD5.
2. Gera uma imagem de preview lado a lado com grid visual (preview_unknown.png).
3. Permite ao usuário associar cada imagem numerada à sua carta real (ex: 1=5d).
4. Extrai a metade superior do canto para template de Rank (bounding box limpa).
5. Extrai a metade inferior do canto para template de Naipe (bounding box limpa).
6. Remove os arquivos processados de unknown_cards/.
7. Atualiza automaticamente o preview.
"""

import cv2
import glob
import hashlib
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import UNKNOWN_CARDS_DIR, TEMPLATES_DIR, RANKS_DIR, SUITS_DIR, SAMPLES_DIR, PROJECT_ROOT
UNKNOWN_DIR = str(UNKNOWN_CARDS_DIR)
TEMPLATES_DIR = str(TEMPLATES_DIR)
RANKS_DIR = str(RANKS_DIR)
SUITS_DIR = str(SUITS_DIR)
PREVIEW_PATH = str(SAMPLES_DIR / "preview_unknown.png")

VALID_RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
VALID_SUITS = {"c", "d", "h", "s"}


def get_unique_unknown_cards(unknown_dir: str = UNKNOWN_DIR) -> List[Dict]:
    """
    Escaneia o diretório de cartas desconhecidas e agrupa duplicatas por hash MD5.
    Retorna uma lista ordenada com os metadados de cada grupo único.
    """
    if not os.path.isdir(unknown_dir):
        return []

    file_paths = sorted(glob.glob(os.path.join(unknown_dir, "*.png")))
    groups: Dict[str, List[str]] = {}

    for path in file_paths:
        try:
            with open(path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h not in groups:
                groups[h] = []
            groups[h].append(path)
        except Exception:
            continue

    unique_cards = []
    for idx, (h, files) in enumerate(groups.items(), 1):
        sample_path = files[0]
        img = cv2.imread(sample_path)
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        unique_cards.append({
            "index": idx,
            "hash": h,
            "files": files,
            "count": len(files),
            "sample_path": sample_path,
            "img": img,
            "width": w_px,
            "height": h_px,
        })

    return unique_cards


def binarize_corner(corner_bgr: np.ndarray) -> np.ndarray:
    """
    Binariza o canto da carta isolando os símbolos (glifos) de Rank e Naipe sem ruído.
    Remove verde do feltro da mesa e isola pixels escuros/vermelhos sobre fundo branco.
    """
    hsv = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)

    # Mascara verde do feltro
    green_mask = cv2.inRange(hsv, np.array([30, 50, 30]), np.array([90, 255, 255]))

    glyph_mask = np.zeros_like(gray)
    glyph_mask[(green_mask == 0) & (gray < 185)] = 255
    return glyph_mask


def clean_glyph_region(binary_crop: np.ndarray) -> np.ndarray:
    """
    Limpa ruídos de borda e isola os componentes conexos principais do glifo,
    retornando o recorte exato com bounding box ajustada aos pixels ativos (255).
    """
    nb, out, stats, _ = cv2.connectedComponentsWithStats(binary_crop, connectivity=8)
    if nb <= 1:
        return np.zeros_like(binary_crop)

    cleaned = np.zeros_like(binary_crop)
    for i in range(1, nb):
        x, y, cw, ch, area = stats[i]
        # Descarta ruídos pontuais (< 10 pixels)
        if area < 10:
            continue
        # Descarta fiapos de borda direita lateral (x >= 26 e largura fina)
        if x >= 26 and cw <= 4:
            continue
        # Descarta fiapos de borda esquerda
        if x == 0 and cw <= 2 and area < 25:
            continue
        cleaned[out == i] = 255

    pts = cv2.findNonZero(cleaned)
    if pts is not None:
        bx, by, bw, bh = cv2.boundingRect(pts)
        return cleaned[by : by + bh, bx : bx + bw]

    return cleaned


def extract_template_glyphs(card_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extrai o Rank e Naipe do canto superior esquerdo da imagem da carta.
    Retorna (rank_glyph, suit_glyph) como imagens binárias 2D com bounding box limpa.
    """
    ih, iw = card_img.shape[:2]
    ch = min(ih, 52)
    cw = min(iw, 36)
    corner = card_img[:ch, :cw]
    bin_c = binarize_corner(corner)

    # Detecta linha de separação horizontal entre Rank (topo) e Naipe (base)
    proj = np.sum(bin_c > 0, axis=1)
    start_r = max(15, int(corner.shape[0] * 0.40))
    end_r = min(int(corner.shape[0] * 0.75), bin_c.shape[0] - 2)
    search_range = range(start_r, end_r)

    if list(search_range):
        split_y = min(search_range, key=lambda y: proj[y])
    else:
        split_y = corner.shape[0] // 2

    top = bin_c[:split_y, :]
    bot = bin_c[split_y:, :]

    rank_glyph = clean_glyph_region(top)
    suit_glyph = clean_glyph_region(bot)

    return rank_glyph, suit_glyph


def generate_preview(unique_cards: List[Dict], output_path: str = PREVIEW_PATH) -> bool:
    """
    Gera a imagem de preview com grid lado a lado mostrando todas as cartas únicas numeradas.
    """
    num_items = len(unique_cards)
    if num_items == 0:
        if os.path.isfile(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False

    cols = min(6, num_items)
    rows = (num_items + cols - 1) // cols

    cell_w, cell_h = 160, 210
    gap = 15
    pad_x, pad_y = 20, 20
    header_h = 75

    img_w = pad_x * 2 + cols * cell_w + (cols - 1) * gap
    img_h = header_h + pad_y + rows * cell_h + (rows - 1) * gap + pad_y

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    canvas[:] = (30, 25, 22)  # Fundo ardósia escuro

    # Faixa superior de cabeçalho
    cv2.rectangle(canvas, (0, 0), (img_w, header_h), (20, 16, 14), -1)
    cv2.line(canvas, (0, header_h), (img_w, header_h), (60, 50, 45), 2)
    cv2.putText(
        canvas,
        "POKER ANALYTICS - CATALOGADOR DE CARTAS DESCONHECIDAS",
        (pad_x, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    total_files = sum(it["count"] for it in unique_cards)
    subtitle = f"Total: {num_items} cartas unicas ({total_files} capturas no total) | Digite ex: 1=5d no terminal"
    cv2.putText(
        canvas,
        subtitle,
        (pad_x, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    for i, it in enumerate(unique_cards):
        r = i // cols
        c = i % cols
        x0 = pad_x + c * (cell_w + gap)
        y0 = header_h + pad_y + r * (cell_h + gap)
        x1 = x0 + cell_w
        y1 = y0 + cell_h

        # Fundo do card
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (45, 38, 32), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 60, 50), 1)

        # Barra do número de identificação
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + 32), (35, 28, 24), -1)
        cv2.line(canvas, (x0, y0 + 32), (x1, y0 + 32), (70, 60, 50), 1)
        badge_text = f"[ #{it['index']} ]"
        cv2.putText(
            canvas,
            badge_text,
            (x0 + 10, y0 + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        count_text = f"{it['count']}x"
        cv2.putText(
            canvas,
            count_text,
            (x1 - 42, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 100),
            1,
            cv2.LINE_AA,
        )

        # Imagem centralizada
        cimg = it["img"]
        ch, cw = cimg.shape[:2]
        max_dw, max_dh = 130, 140
        scale = min(max_dw / cw, max_dh / ch, 1.25)
        dw = int(round(cw * scale))
        dh = int(round(ch * scale))
        resized = cv2.resize(cimg, (dw, dh), interpolation=cv2.INTER_NEAREST)

        cx = x0 + (cell_w - dw) // 2
        cy = (y0 + 36) + ((y1 - 20) - (y0 + 36) - dh) // 2

        # Borda ao redor da carta
        cv2.rectangle(canvas, (cx - 1, cy - 1), (cx + dw, cy + dh), (100, 100, 100), 1)
        canvas[cy : cy + dh, cx : cx + dw] = resized

        # Informação de dimensão
        dim_text = f"{cw}x{ch} px"
        cv2.putText(
            canvas,
            dim_text,
            (x0 + 10, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, canvas)
    return True


def parse_card_spec(card_str: str) -> Tuple[str, str]:
    """
    Valida e divide a representação de uma carta (ex: '5d', '10s', 'Ah') em (rank, naipe).
    """
    s = card_str.strip().lower()
    if len(s) < 2:
        raise ValueError(f"Formato inválido: '{card_str}'. Esperado ex: 5d, Ah, 10s")

    if s.startswith("10"):
        rank = "10"
        suit = s[2:]
    elif s[0] == "t":
        rank = "10"
        suit = s[1:]
    else:
        rank = s[0].upper()
        suit = s[1:]

    if rank not in VALID_RANKS:
        raise ValueError(f"Rank '{rank}' inválido. Válidos: 2..10, J, Q, K, A")
    if suit not in VALID_SUITS:
        raise ValueError(f"Naipe '{suit}' inválido. Válidos: c (paus), d (ouros), h (copas), s (espadas)")

    return rank, suit


def catalog_single_card(item: Dict, rank: str, suit: str) -> bool:
    """
    Processa a carta especificada:
    1. Extrai rank e suit com bounding box limpa.
    2. Salva em assets/templates/ranks/{rank}.png e assets/templates/suits/{suit}.png.
    3. Remove os arquivos duplicados de unknown_cards/.
    """
    os.makedirs(RANKS_DIR, exist_ok=True)
    os.makedirs(SUITS_DIR, exist_ok=True)

    rank_glyph, suit_glyph = extract_template_glyphs(item["img"])

    created_rank = False
    created_suit = False

    if rank_glyph is not None and (rank_glyph > 0).sum() > 0:
        rank_file = os.path.join(RANKS_DIR, f"{rank}.png")
        cv2.imwrite(rank_file, rank_glyph)
        h, w = rank_glyph.shape[:2]
        print(f"  [+] Template de Rank salvo : assets/templates/ranks/{rank}.png ({w}x{h} px)")
        created_rank = True
    else:
        print(f"  [!] Alerta: Glifo de Rank vazio ou não detectado para '{rank}'.")

    if suit_glyph is not None and (suit_glyph > 0).sum() > 0:
        suit_file = os.path.join(SUITS_DIR, f"{suit}.png")
        cv2.imwrite(suit_file, suit_glyph)
        h, w = suit_glyph.shape[:2]
        print(f"  [+] Template de Naipe salvo: assets/templates/suits/{suit}.png ({w}x{h} px)")
        created_suit = True
    else:
        print(f"  [!] Alerta: Glifo de Naipe vazio ou não detectado para '{suit}'.")

    # Remove os arquivos da lista de desconhecidas
    deleted_count = 0
    for fpath in item["files"]:
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
                deleted_count += 1
        except Exception as e:
            print(f"  [!] Erro ao remover {fpath}: {e}")

    print(f"  [-] {deleted_count} arquivo(s) duplicado(s) removido(s) de assets/unknown_cards/.\n")
    return created_rank or created_suit


def delete_single_group(item: Dict) -> None:
    """Remove um grupo de capturas inválidas ou ruídos sem criar templates."""
    deleted_count = 0
    for fpath in item["files"]:
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
                deleted_count += 1
        except Exception as e:
            print(f"  [!] Erro ao remover {fpath}: {e}")
    print(f"  [-] Grupo #{item['index']} descartado ({deleted_count} arquivo(s) removido(s)).\n")


def print_status_table(unique_cards: List[Dict]):
    """Imprime uma listagem formatada das cartas pendentes no terminal."""
    total_files = sum(it["count"] for it in unique_cards)
    print("\n" + "=" * 75)
    print(" CATALOGADOR DE CARTAS DESCONHECIDAS - Poker Analytics")
    print("=" * 75)
    print(f" Total de cartas únicas: {len(unique_cards)}  |  Total de arquivos: {total_files}")
    print(f" Imagem de preview     : preview_unknown.png")
    print("-" * 75)
    print(f" {'ID':<6} | {'ARQUIVOS':<10} | {'TAMANHO':<12} | {'EXEMPLO DE ARQUIVO'}")
    print("-" * 75)
    for it in unique_cards:
        idx_str = f"[ #{it['index']} ]"
        cnt_str = f"{it['count']}x"
        size_str = f"{it['width']}x{it['height']} px"
        sample_name = os.path.basename(it["sample_path"])
        if len(sample_name) > 38:
            sample_name = sample_name[:35] + "..."
        print(f" {idx_str:<6} | {cnt_str:<10} | {size_str:<12} | {sample_name}")
    print("=" * 75)
    print(" Dica: Abra 'preview_unknown.png' para inspecionar visualmente cada número.")
    print(" Digite comandos como '1=5d' ou múltiplos separados por vírgula '1=5d, 2=Ah'.")
    print(" Para descartar ruídos, digite 'del 15'. Para finalizar, digite 'sair'.\n")


def process_user_input(user_input: str, card_map: Dict[int, Dict]) -> Tuple[bool, bool]:
    """
    Processa uma entrada do usuário.
    Retorna (should_exit, had_changes).
    """
    raw = user_input.strip()
    if not raw:
        return False, False

    if raw.lower() in ("sair", "exit", "quit", "q"):
        return True, False

    # Divide comandos separados por vírgula ou espaço
    parts = [p.strip() for p in re.split(r"[,;]+|\s+(?=\d+=|\bdel\b|\brm\b)", raw) if p.strip()]
    had_changes = False

    for part in parts:
        # Comando para deletar grupo (ex: 'del 15' ou 'rm 15')
        del_match = re.match(r"^(?:del|rm|delete)\s+(\d+)$", part, re.IGNORECASE)
        if del_match:
            idx = int(del_match.group(1))
            if idx in card_map:
                delete_single_group(card_map[idx])
                had_changes = True
            else:
                print(f"[ERRO] ID #{idx} não encontrado na lista atual.")
            continue

        # Comando de catalogação (ex: '1=5d' ou '1:5d')
        cat_match = re.match(r"^(\d+)\s*[:=]\s*([0-9a-zA-Z]+)$", part)
        if cat_match:
            idx = int(cat_match.group(1))
            card_code = cat_match.group(2)
            if idx not in card_map:
                print(f"[ERRO] ID #{idx} não encontrado na lista atual.")
                continue
            try:
                rank, suit = parse_card_spec(card_code)
            except ValueError as ve:
                print(f"[ERRO] {ve}")
                continue

            print(f"\n[CATALOGANDO] #{idx} -> Rank '{rank}', Naipe '{suit}'...")
            success = catalog_single_card(card_map[idx], rank, suit)
            if success:
                had_changes = True
            continue

        print(f"[ERRO] Comando não reconhecido: '{part}'. Formato esperado: '1=5d' ou 'del 15'.")

    return False, had_changes


def run_cataloger():
    """Loop principal de interação do catalogador."""
    # Processar argumentos de linha de comando se passados
    args = sys.argv[1:]
    if args:
        if "--preview" in args:
            cards = get_unique_unknown_cards()
            generate_preview(cards)
            print(f"[INFO] Preview gerado em: {PREVIEW_PATH} ({len(cards)} cartas únicas).")
            return

        # Argumentos diretos (ex: python catalog_cards.py 1=5d 2=Ah)
        cards = get_unique_unknown_cards()
        card_map = {it["index"]: it for it in cards}
        for arg in args:
            process_user_input(arg, card_map)
        cards_after = get_unique_unknown_cards()
        generate_preview(cards_after)
        return

    while True:
        unique_cards = get_unique_unknown_cards()
        if not unique_cards:
            print("\n[PARABÉNS] Não há mais cartas desconhecidas em assets/unknown_cards/!")
            if os.path.isfile(PREVIEW_PATH):
                try:
                    os.remove(PREVIEW_PATH)
                except OSError:
                    pass
            break

        generate_preview(unique_cards)
        print_status_table(unique_cards)

        card_map = {it["index"]: it for it in unique_cards}

        try:
            user_input = input("Digite o número da imagem e a carta correspondente (ex: 1=5d, 2=Ah, ou 'sair'): ")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando catalogador.")
            break

        should_exit, had_changes = process_user_input(user_input, card_map)
        if should_exit:
            print("Catalogador encerrado pelo usuário.")
            break


if __name__ == "__main__":
    run_cataloger()

