"""
src/engine/range_model.py
Modelagem de Ranges de Oponentes usando Bayesian Hand Weights.
"""

from typing import List, Tuple, Set, Dict
import itertools
import random

# Cache global das combinações
_ALL_COMBOS = None
_VAL_MAP = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

def get_all_combos(full_deck: List[str]) -> List[Tuple[str, str]]:
    global _ALL_COMBOS
    if _ALL_COMBOS is not None:
        return _ALL_COMBOS
    _ALL_COMBOS = list(itertools.combinations(full_deck, 2))
    return _ALL_COMBOS

def get_chen_score(v1: int, v2: int, suited: bool) -> float:
    if v1 < v2:
        v1, v2 = v2, v1
    def card_points(v: int) -> float:
        if v == 14: return 10.0
        if v == 13: return 8.0
        if v == 12: return 7.0
        if v == 11: return 6.0
        return v / 2.0
    score = card_points(v1)
    is_pair = (v1 == v2)
    if is_pair:
        score = max(5.0, score * 2.0)
    else:
        gap = v1 - v2 - 1
        if gap == 1: score -= 1.0
        elif gap == 2: score -= 2.0
        elif gap == 3: score -= 4.0
        elif gap >= 4: score -= 5.0
    if suited:
        score += 2.0
    if not is_pair and v1 < 12 and (v1 - v2 - 1) <= 1:
        score += 1.0
    return score

class OpponentModel:
    """
    Mantém uma matriz/dicionário de pesos para as 1326 combinações de mãos possíveis.
    O range é degradado/afunilado a cada ação do oponente.
    """
    def __init__(self, full_deck: List[str]):
        self.full_deck = full_deck
        self.weights: Dict[Tuple[str, str], float] = {}
        self.reset()
        
    def reset(self):
        """Inicializa todas as mãos com peso baseado vagamente na força natural (Chen score)."""
        combos = get_all_combos(self.full_deck)
        for c1, c2 in combos:
            r1, s1 = c1[:-1].upper(), c1[-1].lower()
            r2, s2 = c2[:-1].upper(), c2[-1].lower()
            v1 = _VAL_MAP.get(r1, 0)
            v2 = _VAL_MAP.get(r2, 0)
            score = get_chen_score(v1, v2, s1 == s2)
            # Normalizamos o score base
            base_weight = max(0.1, (score + 2.0) / 10.0) 
            self.weights[(c1, c2)] = base_weight

    def apply_action_heuristic(self, pot_size: float, state: str):
        """
        Afunila o range baseado no tamanho do pote (agressividade).
        """
        for combo, w in self.weights.items():
            c1, c2 = combo
            r1, s1 = c1[:-1].upper(), c1[-1].lower()
            r2, s2 = c2[:-1].upper(), c2[-1].lower()
            v1 = _VAL_MAP.get(r1, 0)
            v2 = _VAL_MAP.get(r2, 0)
            is_pair = (v1 == v2)
            is_suited = (s1 == s2)
            
            if state == "PRE_FLOP":
                if pot_size > 30: # 3-bet pot ou grande open raise
                    if not is_pair and v1 < 10 and v2 < 10:
                        self.weights[combo] *= 0.05
                    elif is_pair and v1 >= 10:
                        self.weights[combo] *= 2.0
                    elif v1 >= 13 and v2 >= 12: # AK, KQ, AQ
                        self.weights[combo] *= 1.5
                elif pot_size > 15: # Standard raise
                    if not is_pair and not is_suited and v1 < 9:
                        self.weights[combo] *= 0.2
            else:
                # Pós-flop agressivo
                if pot_size > 50:
                    if not is_pair and v1 < 8 and v2 < 8 and not is_suited:
                        self.weights[combo] *= 0.1

    def apply_postflop_board(self, board_cards: List[str], pot_size: float):
        """
        Atualiza probabilidades baseando-se na textura do board.
        """
        if not board_cards or pot_size < 30:
            return
            
        board_vals = set(_VAL_MAP.get(c[:-1].upper(), 0) for c in board_cards if len(c) >= 2)
        board_suits = [c[-1].lower() for c in board_cards if len(c) >= 2]
        suit_counts = {s: board_suits.count(s) for s in set(board_suits)}
        flush_draw_suit = None
        for s, count in suit_counts.items():
            if count >= 2:
                flush_draw_suit = s
                
        for combo, w in self.weights.items():
            c1, c2 = combo
            v1 = _VAL_MAP.get(c1[:-1].upper(), 0)
            v2 = _VAL_MAP.get(c2[:-1].upper(), 0)
            s1 = c1[-1].lower()
            s2 = c2[-1].lower()
            
            hit_pair = (v1 in board_vals) or (v2 in board_vals)
            hit_pocket_overpair = (v1 == v2) and all(v1 > bv for bv in board_vals)
            hit_flush_draw = flush_draw_suit and (s1 == flush_draw_suit and s2 == flush_draw_suit)
            
            if hit_pair or hit_pocket_overpair or hit_flush_draw:
                self.weights[combo] *= 1.5
            else:
                self.weights[combo] *= 0.5

    def get_weighted_combos(self, dead_cards: Set[str]) -> Tuple[List[Tuple[str, str]], List[float]]:
        """
        Retorna (lista de combos, lista de pesos) para uso no Monte Carlo.
        """
        valid_combos = []
        valid_weights = []
        for combo, w in self.weights.items():
            if w > 0.001 and combo[0] not in dead_cards and combo[1] not in dead_cards:
                valid_combos.append(combo)
                valid_weights.append(w)
                
        return valid_combos, valid_weights

# Fallback para o teste/interface antiga
def get_opponent_range(full_deck: List[str], dead_cards: Set[str], top_percent: float) -> List[Tuple[str, str]]:
    model = OpponentModel(full_deck)
    # se o percent for mt restrito, força via heuristica manual rapida
    if top_percent < 1.0:
        model.apply_action_heuristic(pot_size=100.0, state="PRE_FLOP")
    combos, weights = model.get_weighted_combos(dead_cards)
    
    # Sort by weight and cut off at top_percent
    scored = list(zip(combos, weights))
    scored.sort(key=lambda x: x[1], reverse=True)
    limit = max(1, int(len(scored) * top_percent))
    return [c for c, w in scored[:limit]]
