"""
src/engine/range_model.py
Modelagem de Ranges de Oponentes usando Fórmula de Chen para aproximação de força.
"""

from typing import List, Tuple, Set
import itertools

def get_chen_score(v1: int, v2: int, suited: bool) -> float:
    """
    Calcula a pontuação da mão usando a Fórmula de Chen.
    v1, v2: valores das cartas de 2 a 14 (A=14). Assume-se v1 >= v2.
    """
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
        # Penalidade por gap
        gap = v1 - v2 - 1
        if gap == 1: score -= 1.0
        elif gap == 2: score -= 2.0
        elif gap == 3: score -= 4.0
        elif gap >= 4: score -= 5.0
        
    if suited:
        score += 2.0
        
    # Bônus de conectividade para cartas baixas
    if not is_pair and v1 < 12 and (v1 - v2 - 1) <= 1:
        score += 1.0
        
    return score

# Cache global das combinações ordenadas
_RANKED_HANDS = None

def get_ranked_hands(full_deck: List[str]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Retorna todas as combinações de 2 cartas (1326 totais) ordenadas pela força de Chen.
    """
    global _RANKED_HANDS
    if _RANKED_HANDS is not None:
        return _RANKED_HANDS
        
    val_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    
    scored = []
    for c1, c2 in itertools.combinations(full_deck, 2):
        r1, s1 = c1[:-1].upper(), c1[-1].lower()
        r2, s2 = c2[:-1].upper(), c2[-1].lower()
        
        v1 = val_map.get(r1, 0)
        v2 = val_map.get(r2, 0)
        
        score = get_chen_score(v1, v2, s1 == s2)
        scored.append((score, (c1, c2)))
        
    # Ordena decrescente pelo score
    scored.sort(key=lambda x: x[0], reverse=True)
    _RANKED_HANDS = scored
    return _RANKED_HANDS

def get_opponent_range(full_deck: List[str], dead_cards: Set[str], top_percent: float) -> List[Tuple[str, str]]:
    """
    Retorna as combinações de 2 cartas que representam os top X% do range do oponente,
    excluindo cartas que já estão na mesa ou na mão do hero (dead_cards).
    """
    ranked = get_ranked_hands(full_deck)
    
    # Filtra dead cards e mantém a ordem
    valid_combos = []
    for score, combo in ranked:
        c1, c2 = combo
        if c1 not in dead_cards and c2 not in dead_cards:
            valid_combos.append(combo)
            
    if not valid_combos:
        return []
        
    top_percent = max(0.01, min(1.0, top_percent))
    limit = max(1, int(len(valid_combos) * top_percent))
    
    return valid_combos[:limit]
