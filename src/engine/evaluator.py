"""
src/engine/evaluator.py
Motor de Avaliação de Mãos, Detecção de Projetos (Draws) e Simulação de Monte Carlo.
"""

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Permite execução direta como script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Constantes e Mapeamentos do Baralho
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["s", "h", "d", "c"]  # Spades, Hearts, Diamonds, Clubs

RANK_VALUES: Dict[str, int] = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

from src.engine.range_model import get_opponent_range

VALUE_TO_RANK: Dict[int, str] = {v: r for r, v in RANK_VALUES.items() if r != "T"}
VALUE_TO_RANK[10] = "10"

FULL_DECK: List[str] = [r + s for r in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] for s in SUITS]

HAND_CATEGORIES: Dict[int, str] = {
    8: "Straight Flush",
    7: "Four of a Kind (Quadra)",
    6: "Full House",
    5: "Flush (Cor)",
    4: "Straight (Sequência)",
    3: "Three of a Kind (Trinca)",
    2: "Two Pair (Dois Pares)",
    1: "One Pair (Um Par)",
    0: "High Card (Carta Alta)"
}


def parse_card(card: str) -> Tuple[int, str]:
    """Converte string de carta (ex: 'Ah', '10s') em tupla (valor_inteiro, naipe)."""
    card = card.strip()
    if not card or card.lower() in ("desconhecida", "vazio", "none"):
        raise ValueError(f"Carta inválida para parsing: '{card}'")
    if card.startswith("10"):
        return 10, card[2:].lower()
    rank_char = card[0].upper()
    if rank_char not in RANK_VALUES:
        raise ValueError(f"Rank desconhecido '{rank_char}' na carta: '{card}'")
    return RANK_VALUES[rank_char], card[1:].lower()


def check_straight(unique_ranks_desc: List[int]) -> int:
    """
    Verifica se uma lista ordenada decrescente de ranks únicos contém uma sequência (Straight).
    Retorna o valor da maior carta da sequência, ou 0 se não houver.
    """
    n = len(unique_ranks_desc)
    for i in range(n - 4):
        if unique_ranks_desc[i] - unique_ranks_desc[i + 4] == 4:
            return unique_ranks_desc[i]

    # Regra especial: Ace-to-Five (Wheel / Broadway inferior: A-2-3-4-5)
    if 14 in unique_ranks_desc and {5, 4, 3, 2}.issubset(unique_ranks_desc):
        return 5

    return 0


def evaluate_7_cards(cards: List[str]) -> Tuple[int, ...]:
    """
    Avalia a melhor mão de 5 cartas a partir de 5, 6 ou 7 cartas disponíveis.
    """
    valid_cards = [c for c in cards if c and c.lower() not in ("desconhecida", "vazio", "none", "--")]
    if len(valid_cards) < 5:
        raise ValueError(f"Mínimo de 5 cartas válidas exigido para avaliação. Recebido: {cards}")

    parsed = [parse_card(c) for c in valid_cards]

    # 1. Agrupamento por Naipe (para Flush / Straight Flush)
    suits_dict: Dict[str, List[int]] = {}
    for r_val, s in parsed:
        suits_dict.setdefault(s, []).append(r_val)

    flush_suit = None
    for s, r_list in suits_dict.items():
        if len(r_list) >= 5:
            flush_suit = s
            break

    # 2. Straight Flush / Royal Flush
    if flush_suit:
        flush_ranks = sorted(set(suits_dict[flush_suit]), reverse=True)
        sf_high = check_straight(flush_ranks)
        if sf_high > 0:
            return (8, sf_high)

    # 3. Contagem de Frequência de Ranks
    rank_counts: Dict[int, int] = {}
    for r_val, _ in parsed:
        rank_counts[r_val] = rank_counts.get(r_val, 0) + 1

    counts_sorted = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    top_r, top_c = counts_sorted[0]
    unique_ranks = sorted(rank_counts.keys(), reverse=True)

    # 4. Four of a Kind (Quadra)
    if top_c == 4:
        kickers = [r for r in unique_ranks if r != top_r][:1]
        return (7, top_r, *kickers)

    # 5. Full House
    if top_c == 3 and len(counts_sorted) > 1 and counts_sorted[1][1] >= 2:
        second_r = counts_sorted[1][0]
        return (6, top_r, second_r)

    # 6. Flush (Cor)
    if flush_suit:
        top_flush_ranks = sorted(suits_dict[flush_suit], reverse=True)[:5]
        return (5, *top_flush_ranks)

    # 7. Straight (Sequência)
    straight_high = check_straight(unique_ranks)
    if straight_high > 0:
        return (4, straight_high)

    # 8. Three of a Kind (Trinca)
    if top_c == 3:
        kickers = [r for r in unique_ranks if r != top_r][:2]
        return (3, top_r, *kickers)

    # 9. Two Pair (Dois Pares) ou One Pair (Um Par)
    if top_c == 2:
        if counts_sorted[1][1] == 2:
            r1 = top_r
            r2 = counts_sorted[1][0]
            kickers = [r for r in unique_ranks if r != r1 and r != r2][:1]
            return (2, r1, r2, *kickers)
        else:
            r1 = top_r
            kickers = [r for r in unique_ranks if r != r1][:3]
            return (1, r1, *kickers)

    # 10. High Card (Carta Alta)
    return (0, *unique_ranks[:5])


def get_hand_category_name(score: Tuple[int, ...]) -> str:
    """Retorna o nome amigável da categoria da mão."""
    category_id = score[0]
    return HAND_CATEGORIES.get(category_id, "Desconhecida")


def calculate_equity(
    hero_cards: List[str],
    board_cards: List[str],
    num_opponents: int = 4,
    iterations: int = 10000,
    opp_range_percent: float = 1.0
) -> Tuple[float, float, float, float, Tuple[int, ...]]:
    """
    Calcula a probabilidade de Vitória, Empate e Derrota via Simulação de Monte Carlo.
    """
    valid_hero = [c for c in hero_cards if c and c.lower() not in ("desconhecida", "vazio", "none", "--")]
    valid_board = [c for c in board_cards if c and c.lower() not in ("desconhecida", "vazio", "none", "--")]

    if len(valid_hero) != 2:
        raise ValueError(f"Hero deve possuir exatamente 2 cartas válidas. Recebido: {hero_cards}")

    num_opponents = max(1, min(int(num_opponents), 8))
    dead_cards = set(valid_hero + valid_board)
    available_deck = [c for c in FULL_DECK if c not in dead_cards]
    cards_needed_board = 5 - len(valid_board)

    hero_current_score = evaluate_7_cards(valid_hero + valid_board) if len(valid_hero + valid_board) >= 5 else (0, 0)

    wins = 0
    ties = 0
    losses = 0
    t_start = time.time()

    opp_combos = []
    if opp_range_percent < 1.0:
        opp_combos = get_opponent_range(FULL_DECK, dead_cards, opp_range_percent)

    for _ in range(iterations):
        drawn_board = random.sample(available_deck, cards_needed_board)
        sim_board = valid_board + drawn_board

        hero_score = evaluate_7_cards(valid_hero + sim_board)

        hero_tied = False
        hero_lost = False

        if opp_combos:
            used_cards = set(drawn_board)
            for _ in range(num_opponents):
                for _ in range(10): # retry limit
                    opp_cards = random.choice(opp_combos)
                    if opp_cards[0] not in used_cards and opp_cards[1] not in used_cards:
                        used_cards.add(opp_cards[0])
                        used_cards.add(opp_cards[1])
                        break
                else:
                    avail_opp = [c for c in available_deck if c not in used_cards]
                    opp_cards = tuple(random.sample(avail_opp, 2))
                    used_cards.add(opp_cards[0])
                    used_cards.add(opp_cards[1])

                opp_score = evaluate_7_cards(list(opp_cards) + sim_board)
                if opp_score > hero_score:
                    hero_lost = True
                    break
                elif opp_score == hero_score:
                    hero_tied = True
        else:
            avail_opp = [c for c in available_deck if c not in drawn_board]
            drawn_opps = random.sample(avail_opp, num_opponents * 2)
            idx_offset = 0
            for _ in range(num_opponents):
                opp_cards = drawn_opps[idx_offset : idx_offset + 2]
                idx_offset += 2
                opp_score = evaluate_7_cards(opp_cards + sim_board)

                if opp_score > hero_score:
                    hero_lost = True
                    break
                elif opp_score == hero_score:
                    hero_tied = True

        if hero_lost:
            losses += 1
        elif hero_tied:
            ties += 1
        else:
            wins += 1

    elapsed_time = time.time() - t_start
    p_win = (wins / iterations) * 100.0
    p_tie = (ties / iterations) * 100.0
    p_lose = (losses / iterations) * 100.0

    equity = round(p_win + (p_tie / 2.0), 2)
    return round(p_win, 2), round(p_tie, 2), round(p_lose, 2), equity, round(elapsed_time, 3)


def detect_draws(hero_cards: List[str], board_cards: List[str]) -> Tuple[int, str]:
    """
    Analisa os projetos (draws) no Flop e Turn.
    Retorna (número de Outs, Nome do Projeto).
    """
    if len(board_cards) not in [3, 4]:
        return 0, ""

    all_cards = hero_cards + board_cards
    val_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }

    suits: Dict[str, int] = {}
    ranks = set()
    for c in all_cards:
        if not c or len(c) < 2 or c.lower() in ("desconhecida", "vazio", "none"):
            continue
        r, s = c[:-1].upper(), c[-1].lower()
        if r not in val_map:
            continue
        v = val_map[r]
        ranks.add(v)
        suits[s] = suits.get(s, 0) + 1

    outs = 0
    draw_type = []

    # Flush draw
    if any(count == 4 for count in suits.values()):
        outs += 9
        draw_type.append("Flush Draw (9 Outs)")

    straight_outs = set()
    for v in range(2, 11):
        needed = set(range(v, v + 5))
        if v == 2:
            needed = {14, 2, 3, 4, 5}

        overlap = needed.intersection(ranks)
        if len(overlap) == 4:
            missing = needed - overlap
            straight_outs.add(missing.pop())

    if len(straight_outs) > 0:
        if len(straight_outs) >= 2:
            draw_type.append("OESD / Open-Ended (8 Outs)")
            outs += 8
        else:
            draw_type.append("Gutshot (4 Outs)")
            outs += 4

    if "Flush Draw (9 Outs)" in draw_type and len(straight_outs) > 0:
        outs -= len(straight_outs)

    if not draw_type:
        return 0, ""

    return outs, " + ".join(draw_type)
