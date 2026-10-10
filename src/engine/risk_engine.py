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
    # Probabilidades vêm como porcentagem (0-100), dividimos por 100 para a fórmula do EV
    return ((p_win / 100.0) * pot_size) - ((p_lose / 100.0) * bet_to_call) + ((p_tie / 100.0) * (pot_size / 2.0))



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
    has_pair_or_better: bool,
    state: str = "PRE_FLOP"
) -> float:
    """
    Camada 2 - Estratégia: Calcula um Score de Decisão Multidimensional (0 a 100+).
    Pesos Iniciais: Equity (40%), Pot Odds (20%), SPR (20%), Board Texture (10%), Pre-flop Tier (10%)
    """
    score = 0.0
    
    # 1. Equity & Pot Odds (Math Base)
    # Pré-flop as equidades são comprimidas (mesmo o AA tem "apenas" 80% HU e 50% vs 3).
    # Então multiplicamos a vantagem de equidade por 2.5 no pre-flop para compensar.
    eq_adv = (equity - pot_odds)
    if state == "PRE_FLOP":
        eq_adv *= 2.5

    if equity > pot_odds:
        score += 30.0 + eq_adv
    elif pot_odds > 0:
        score += 30.0 * (equity / pot_odds)
        
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
            
    # 4. Preflop Tier Strength (Ajustado para VPIP mais natural em mesas casuais)
    if tier == 1:
        score += 20.0
    elif tier == 2:
        score += 12.0
    elif tier == 3:
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
    draw_name: str = "Nenhum",
    loose_mode: bool = False
) -> Dict[str, Any]:
    """
    Motor de Risco Multidimensional. Retorna a decisão (FOLD/CHECK/CALL/RAISE/ALL-IN).
    """
    equity = p_win + (p_tie / 2.0)
    pot_odds = (bet_to_call / (pot_size + bet_to_call)) * 100.0 if (pot_size + bet_to_call) > 0 else 0.0
    ev = ((p_win / 100.0) * pot_size) - ((p_lose / 100.0) * bet_to_call) + ((p_tie / 100.0) * (pot_size / 2.0))
    
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
            
    # Issue #9: Blueprint Strategy (Camada 1) e Subgame Solving (Camada 2)
    if state == "PRE_FLOP":
        try:
            from src.engine.blueprint import PreflopBlueprint
            strategy = PreflopBlueprint.get_strategy(tier, bet_to_call, spr)
            chosen_action = PreflopBlueprint.sample_action(strategy)
            
            action = "FOLD"
            recommended_amount = 0.0
            
            if chosen_action == "FOLD":
                action = "FOLD" if bet_to_call > 0 else "CHECK"
            elif chosen_action == "CALL":
                action = "CALL" if bet_to_call > 0 else "CHECK"
                recommended_amount = bet_to_call
            elif chosen_action.startswith("BET_"):
                pct = float(chosen_action.split("_")[1]) / 100.0
                action = "RAISE" if bet_to_call > 0 else "BET"
                recommended_amount = min(hero_stack, round(pot_size * pct, 1))
            elif chosen_action == "ALL_IN":
                action = "ALL-IN"
                recommended_amount = hero_stack
                
            # Traduz chaves da estratégia para strings reais no painel
            display_strategy = {}
            for a, p in strategy.items():
                disp_a = a
                if a == "FOLD": disp_a = "FOLD" if bet_to_call > 0 else "CHECK"
                elif a == "CALL": disp_a = "CALL" if bet_to_call > 0 else "CHECK"
                elif a.startswith("BET_"): disp_a = "RAISE" if bet_to_call > 0 else "BET"
                elif a == "ALL_IN": disp_a = "ALL-IN"
                display_strategy[disp_a] = display_strategy.get(disp_a, 0.0) + p

            strat_str = ", ".join([f"{a}:{p:.2f}" for a, p in display_strategy.items() if p > 0.01])
            reason = f"Blueprint Strategy (Tier {tier}) [{strat_str}] -> Escolhido: {action}"
            
            if action == "FOLD" and ev > 0 and bet_to_call > 0:
                reason = f"EV marginal ({ev:+.1f}), mas Blueprint (Tier {tier}) indica FOLD por alta variância/rake. [{strat_str}]"
            
            return {
                "action": action,
                "ev": ev,
                "pot_odds": pot_odds,
                "equity": equity,
                "bet_to_call": bet_to_call,
                "pot_size": pot_size,
                "recommended_amount": recommended_amount,
                "reason": reason,
                "cfr_strategy": strategy
            }
        except ImportError:
            pass
    else:
        # PÓS-FLOP: Subgame Solving com CFR+ (Camada 2)
        try:
            from src.engine.game_tree import GameNode, GameTreeBuilder
            from src.engine.cfr_solver import CFRSolver
            import random
            
            root_node = GameNode(
                player=0,
                pot_size=pot_size,
                hero_stack=hero_stack,
                villain_stack=hero_stack, # Assumindo stack efetivo espelhado temporariamente
                board_cards=board_cards if board_cards else [],
                current_street=state,
                bet_to_call=bet_to_call
            )
            
            # Constrói a árvore de sub-jogo
            builder = GameTreeBuilder(max_depth_per_street=1)
            root_node = builder.build_tree(root_node)
            
            if hero_cards:
                solver = CFRSolver()
                cfr_strategy = solver.solve(root_node, hero_cards, equity, iterations=30)
                
                if cfr_strategy:
                    # Seleciona a ação com a maior probabilidade (Argmax) para UX consistente
                    chosen_action = max(cfr_strategy.items(), key=lambda x: x[1])[0]
                    
                    action = "FOLD"
                    recommended_amount = 0.0
                    
                    if chosen_action == "FOLD":
                        action = "FOLD" if bet_to_call > 0 else "CHECK"
                    elif chosen_action == "CALL":
                        action = "CALL" if bet_to_call > 0 else "CHECK"
                        recommended_amount = bet_to_call
                    elif chosen_action.startswith("BET_"):
                        pct = float(chosen_action.split("_")[1]) / 100.0
                        action = "RAISE" if bet_to_call > 0 else "BET"
                        recommended_amount = min(hero_stack, round(pot_size * pct, 1))
                    elif chosen_action == "ALL_IN":
                        action = "ALL-IN"
                        recommended_amount = hero_stack
                        
                    # Traduz chaves da estratégia para strings reais no painel
                    display_strategy = {}
                    for a, p in cfr_strategy.items():
                        disp_a = a
                        if a == "FOLD": disp_a = "FOLD" if bet_to_call > 0 else "CHECK"
                        elif a == "CALL": disp_a = "CALL" if bet_to_call > 0 else "CHECK"
                        elif a.startswith("BET_"): disp_a = "RAISE" if bet_to_call > 0 else "BET"
                        elif a == "ALL_IN": disp_a = "ALL-IN"
                        
                        # Soma probabilidades se colidirem (ex: FOLD->CHECK e CALL->CHECK)
                        display_strategy[disp_a] = display_strategy.get(disp_a, 0.0) + p

                    strat_str = ", ".join([f"{a}:{p:.2f}" for a, p in display_strategy.items() if p > 0.01])
                    reason = f"CFR+ Subgame Solver [{strat_str}] -> Escolhido: {action}"
                    
                    return {
                        "action": action,
                        "ev": ev,
                        "pot_odds": pot_odds,
                        "equity": equity,
                        "bet_to_call": bet_to_call,
                        "pot_size": pot_size,
                        "recommended_amount": recommended_amount,
                        "reason": reason,
                        "cfr_strategy": cfr_strategy
                    }
        except Exception as e:
            print(f"CFR Solver falhou, caindo para heurística: {e}")
            
            
    # Camada 2: Cálculo do Score Multidimensional (Confiança/Estratégia) Fallback
    decision_score = calculate_decision_score(
        equity=equity, pot_odds=pot_odds, ev=ev, spr=spr, 
        texture=texture, tier=tier, bet_ratio=bet_ratio, 
        has_pair_or_better=has_pair_or_better,
        state=state
    )

    action = "FOLD"
    recommended_amount = 0.0

    
    if bet_to_call == 0:
        if decision_score >= 60.0:
            action = "BET"
            recommended_amount = max(1.0, round(pot_size * 0.5, 1))
            reason = f"Mesa em check. Aposta por valor (Score de Confiança {decision_score:.1f})."
        else:
            action = "CHECK"
            reason = f"Mesa em check. Controle de pote e SPR (Score de Confiança {decision_score:.1f})."
    else:
        if ev < 0:
            margin = pot_odds - equity
            
            # Issue #3: Implied Odds Tolerance (Pre-flop)
            # Permite pagar apostas baixas no pré-flop com mãos de potencial (Ax suited, pares) mesmo se o EV for negativo
            implied_odds_tolerance = 3.5
            is_cheap_call = bet_to_call <= (hero_stack * 0.05)
            
            if is_cheap_call and (decision_score >= 50.0 or tier <= 3):  # Mãos tier 1-3 ganham bônus
                implied_odds_tolerance = 15.0  # Tolera até 15% de desvantagem matemática por causa do potencial
            
            if margin <= implied_odds_tolerance and is_cheap_call and (decision_score >= 50.0 or tier <= 3):
                action = "CALL"
                recommended_amount = bet_to_call
                reason = f"Borderline negativo ({ev:+.1f}), mas CALL por Implied Odds pré-flop (Aposta baixa)."
            elif margin <= 3.5:
                action = "FOLD"
                reason = f"EV levemente negativo ({ev:+.1f}). Decisão marginal / Borderline FOLD."
            else:
                action = "FOLD"
                reason = f"EV matemático negativo ({ev:+.1f}). Matemática não justifica o Call."
        else:
            call_threshold = 35.0 if loose_mode else 45.0
            
            if decision_score < call_threshold:
                action = "FOLD"
                reason = f"EV bruto marginal ({ev:+.1f}), porém FOLD estratégico (Baixa Confiança: {decision_score:.1f})."
            elif decision_score >= 75.0:
                action = "RAISE"
                recommended_amount = min(hero_stack, round(bet_to_call * 2.5, 1))
                if hero_stack <= recommended_amount * 1.5:
                    action = "ALL-IN"
                    recommended_amount = hero_stack
                reason = f"EV positivo ({ev:+.1f}). Raise para extrair valor/proteção (Confiança: {decision_score:.1f})."
            else:
                action = "CALL"
                recommended_amount = bet_to_call
                reason = f"EV positivo ({ev:+.1f}). Call estratégico justificado (Confiança: {decision_score:.1f})."

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
