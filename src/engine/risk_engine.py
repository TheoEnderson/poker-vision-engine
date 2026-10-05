"""
src/engine/risk_engine.py
Motor de Decisão Multidimensional (Math, Strategy, Risk).
Calcula Pot Odds, Valor Esperado (EV), SPR, Textura do Board e emite um Decision Score.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.engine.evaluator import evaluate_7_cards
    from src.engine.preflop_tier import get_preflop_tier
except ImportError:
    from evaluator import evaluate_7_cards
    from preflop_tier import get_preflop_tier

def calculate_pot_odds(bet_to_call: float, pot_size: float) -> float:
    return (bet_to_call / (pot_size + bet_to_call)) * 100.0 if (pot_size + bet_to_call) > 0 else 0.0

def calculate_ev(p_win: float, p_lose: float, p_tie: float, pot_size: float, bet_to_call: float) -> float:
    return (p_win * pot_size) - (p_lose * bet_to_call) + (p_tie * (pot_size / 2.0))



def analyze_board_texture(board_cards: List[str]) -> Dict[str, bool]:
    """Analisa a textura do board para ajustar agressividade e descontos de equity."""
    texture = {
        "paired": False,
        "monotone": False,
        "connected": False,
        "dry": True
    }
    
    if not board_cards or len(board_cards) < 3:
        return texture
        
    val_map = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
    
    try:
        ranks = sorted([val_map[c[:-1].upper()] for c in board_cards if len(c) >= 2])
        suits = [c[-1].lower() for c in board_cards if len(c) >= 2]
        
        # Paired
        if len(set(ranks)) < len(ranks):
            texture["paired"] = True
            texture["dry"] = False
            
        # Monotone / Flush-heavy
        suit_counts = {s: suits.count(s) for s in set(suits)}
        if max(suit_counts.values(), default=0) >= 3:
            texture["monotone"] = True
            texture["dry"] = False
            
        # Connected
        gaps = 0
        for i in range(len(ranks) - 1):
            if ranks[i+1] - ranks[i] <= 2 and ranks[i+1] != ranks[i]:
                gaps += 1
        if gaps >= 2:
            texture["connected"] = True
            texture["dry"] = False
            
    except Exception:
        pass
        
    return texture


def calculate_decision_score(
    equity: float, 
    pot_odds: float, 
    ev: float, 
    spr: float, 
    texture: Dict[str, bool],
    tier: int,
    bet_ratio: float,
    has_pair_or_better: bool
) -> float:
    """
    Camada 2 - Estratégia: Calcula um Score de Decisão Multidimensional (0 a 100+).
    Pesos Iniciais: Equity (40%), Pot Odds (20%), SPR (20%), Board Texture (10%), Pre-flop Tier (10%)
    """
    score = 0.0
    
    # 1. Equity & Pot Odds (Math Base)
    if equity > pot_odds:
        score += 30.0 + (equity - pot_odds)  # EV positivo dá base sólida
    elif pot_odds > 0:
        score += 30.0 * (equity / pot_odds)  # Proporcional se for EV negativo
        
    # 2. SPR (Stack to Pot Ratio)
    # Se SPR baixo (comprometido), mãos prontas/fortes ganham muito valor. Se SPR alto, joga com cautela.
    if spr < 3.0 and has_pair_or_better:
        score += 20.0
    elif spr >= 3.0 and has_pair_or_better:
        score += 10.0
        
    # 3. Board Texture
    # Desconta score se o board é perigoso (monotone/conected) e não temos equidade massiva
    if texture["monotone"] or texture["connected"]:
        if equity < 60.0:
            score -= 15.0
            
    # 4. Preflop Tier Strength
    if tier == 1:
        score += 15.0
    elif tier == 2:
        score += 5.0
    elif tier == 4:
        score -= 10.0
        
    # 5. Risk Penalty (Aposta > 40% do stack)
    if bet_ratio > 0.40 and not has_pair_or_better and tier > 1:
        score -= 30.0
        
    return score


def make_decision(
    p_win: float,
    p_lose: float,
    p_tie: float,
    pot_size: float,
    bet_to_call: float,
    hero_stack: float,
    state: str = "PRE_FLOP",
    hero_cards: Optional[List[str]] = None,
    board_cards: Optional[List[str]] = None,
    outs: int = 0,
    draw_name: str = "Nenhum"
) -> Dict[str, Any]:
    """
    Motor de Risco Multidimensional. Retorna a decisão (FOLD/CHECK/CALL/RAISE/ALL-IN).
    """
    equity = p_win + (p_tie / 2.0)
    pot_odds = (bet_to_call / (pot_size + bet_to_call)) * 100.0 if (pot_size + bet_to_call) > 0 else 0.0
    ev = (p_win * pot_size) - (p_lose * bet_to_call) + (p_tie * (pot_size / 2.0))
    
    # Camada 1: Matemática (SPR)
    spr = hero_stack / pot_size if pot_size > 0 else 10.0
    bet_ratio = bet_to_call / hero_stack if hero_stack > 0 else 1.0
    
    # Identificação de Força da Mão e Textura
    tier = get_preflop_tier(hero_cards) if hero_cards else 4
    texture = analyze_board_texture(board_cards) if board_cards else {"paired": False, "monotone": False, "connected": False, "dry": True}
    
    has_pair_or_better = False
    if hero_cards and board_cards:
        score_cards = evaluate_7_cards(hero_cards + board_cards)
        if score_cards[0] >= 1: # Pelo menos um par
            has_pair_or_better = True
            
    # Premium Pre-flop bypass
    if state == "PRE_FLOP" and tier == 1:
        if bet_to_call > 0 and bet_ratio < 0.25:
            return {
                "action": "RAISE", "ev": ev, "pot_odds": pot_odds, "equity": equity,
                "bet_to_call": bet_to_call, "pot_size": pot_size, 
                "recommended_amount": min(hero_stack, round(bet_to_call * 3.0, 1)),
                "reason": "Mão Premium (Tier 1). RAISE/3-BET obrigatório para extrair valor."
            }

    # Camada 2: Cálculo do Score Multidimensional
    decision_score = calculate_decision_score(
        equity=equity, pot_odds=pot_odds, ev=ev, spr=spr, 
        texture=texture, tier=tier, bet_ratio=bet_ratio, 
        has_pair_or_better=has_pair_or_better
    )

    action = "FOLD"
    recommended_amount = 0.0
    
    if bet_to_call == 0:
        if decision_score >= 60.0:
            action = "BET"
            recommended_amount = max(1.0, round(pot_size * 0.5, 1))
            reason = f"Mesa em check, Board {('Seco' if texture['dry'] else 'Perigoso')}, Score alto ({decision_score:.1f}). Value Bet."
        else:
            action = "CHECK"
            reason = f"Mesa em check, Score moderado ({decision_score:.1f}). Controle de pote e SPR ({spr:.1f})."
    else:
        if decision_score >= 75.0:
            action = "RAISE"
            recommended_amount = min(hero_stack, round(bet_to_call * 2.5, 1))
            if hero_stack <= recommended_amount * 1.5:
                action = "ALL-IN"
                recommended_amount = hero_stack
            reason = f"Score dominante ({decision_score:.1f} >= 75). Board analisado. Extraindo valor máximo."
        elif decision_score >= 45.0:
            action = "CALL"
            recommended_amount = bet_to_call
            reason = f"Score moderado/bom ({decision_score:.1f}). SPR atual ({spr:.1f}). CALL justificável matematicamente."
        else:
            action = "FOLD"
            reason = f"Score baixo ({decision_score:.1f} < 45). Board {'Seco' if texture['dry'] else 'Conectado/Perigoso'}, EV {ev:+.1f}. Descartando mão."

    return {
        "action": action,
        "ev": ev,
        "pot_odds": pot_odds,
        "equity": equity,
        "bet_to_call": bet_to_call,
        "pot_size": pot_size,
        "recommended_amount": recommended_amount,
        "reason": reason
    }
