import re

with open("src/vision/detector.py", "r") as f:
    content = f.read()

# Pattern for detect_board function
pattern = re.compile(r"def detect_board\(.*?return detected_board", re.DOTALL)

replacement = """def detect_board(
    board_input: Union[str, Path, np.ndarray] = "mesa_6.png",
    threshold: float = DEFAULT_RANK_THRESHOLD,
    verbose: bool = True,
    num_slots: int = 5,
    board_coords: Optional[Dict[str, int]] = None
) -> List[str]:
    \"\"\"Detecta dinamicamente as cartas do board comunitário na imagem da mesa usando contornos.\"\"\"
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

    return detected_board"""

new_content = pattern.sub(replacement, content)
with open("src/vision/detector.py", "w") as f:
    f.write(new_content)
print("detector.py patched successfully")
