"""
src/engine/preflop_tier.py
Classificador heurístico de força de mãos iniciais (Pré-flop) em Tiers de 1 a 4.
"""

from typing import List


def get_preflop_tier(hero_cards: List[str]) -> int:
    """
    Classifica a força inicial (Pré-flop) da mão do Hero em Tiers de 1 a 4.
    
    Tier 1 (Monstros): AA, KK, QQ, JJ, AKs, AKo
    Tier 2 (Fortes): TT, 99, 88, AQs, AJs, KQs, AQo
    Tier 3 (Especulativas/Posição): 77-22, T9s, 98s, 87s, 76s, A2s-A9s
    Tier 4 (Trash): Disconnected/low hands (ex: 74o, J2o)
    """
    if len(hero_cards) != 2:
        return 4
        
    c1, c2 = hero_cards[0].strip(), hero_cards[1].strip()
    if len(c1) < 2 or len(c2) < 2:
        return 4
        
    r1, s1 = c1[:-1].upper(), c1[-1].lower()
    r2, s2 = c2[:-1].upper(), c2[-1].lower()
    
    val_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    
    if r1 not in val_map or r2 not in val_map:
        return 4
        
    v1 = val_map[r1]
    v2 = val_map[r2]
    
    if v1 < v2:
        v1, v2 = v2, v1  # Garante v1 >= v2
        
    suited = (s1 == s2)
    is_pair = (v1 == v2)
    
    # Tier 1
    if is_pair and v1 >= 11:
        return 1  # JJ+
    if v1 == 14 and v2 == 13:
        return 1  # AKs, AKo
    
    # Tier 2
    if is_pair and 8 <= v1 <= 10:
        return 2  # 88, 99, TT
    if v1 == 14 and v2 == 12:
        return 2  # AQs, AQo
    if suited and v1 == 14 and v2 == 11:
        return 2  # AJs
    if suited and v1 == 13 and v2 == 12:
        return 2  # KQs
    
    # Tier 3
    if is_pair and 2 <= v1 <= 7:
        return 3  # 22-77
    if suited and v1 == 14 and v2 <= 9:
        return 3  # A2s-A9s
    if suited and v1 == 13 and v2 <= 11:
        return 3  # K2s-KJs
    if suited and v1 == 12 and v2 <= 10:
        return 3  # Q2s-QTs
    if suited and v1 == v2 + 1:
        return 3  # Todos os conectores do mesmo naipe (ex: 43s, 54s, etc)
    if suited and v1 == v2 + 2:
        return 3  # Todos os 1-gappers do mesmo naipe (ex: 53s, 64s, etc)
    if suited and v1 == v2 + 3 and v1 >= 7:
        return 3  # 2-gappers do mesmo naipe um pouco maiores (ex: 74s+)
    
    # Tier 4
    return 4
