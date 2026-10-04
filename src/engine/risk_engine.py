"""
src/engine/risk_engine.py
Motor de Análise de Risco, Cálculo de Pot Odds, Valor Esperado (EV) e Tomada de Decisão Tática.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Permite execução direta como script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.engine.evaluator import evaluate_7_cards
    from src.engine.preflop_tier import get_preflop_tier
    from src.config import (
        HEAVY_BET_STACK_RATIO,
        MICRO_BET_STACK_RATIO,
        PREFLOP_SPECULATIVE_BET_RATIO,
    )
except ImportError:
    from evaluator import evaluate_7_cards
    from preflop_tier import get_preflop_tier
    HEAVY_BET_STACK_RATIO = 0.40
    MICRO_BET_STACK_RATIO = 0.02
    PREFLOP_SPECULATIVE_BET_RATIO = 0.025


def is_speculative_hand(hero_cards: List[str]) -> bool:
    """Verifica se a mão inicial possui potencial estrutural para pagar pequenas apostas (Implied Odds)."""
    if not hero_cards or len(hero_cards) != 2:
        return False
    c1, c2 = hero_cards[0].strip(), hero_cards[1].strip()
    if len(c1) < 2 or len(c2) < 2:
        return False

    r1, s1 = c1[:-1].upper(), c1[-1].lower()
    r2, s2 = c2[:-1].upper(), c2[-1].lower()

    val_map = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    if r1 not in val_map or r2 not in val_map:
        return False
    v1, v2 = val_map[r1], val_map[r2]

    if v1 < v2:
        v1, v2 = v2, v1

    suited = (s1 == s2)
    connected = (v1 - v2 <= 2)
    has_broadway = (v1 >= 12 or v2 >= 12)

    return suited or connected or has_broadway


def calculate_pot_odds(pot_size: float, bet_to_call: float) -> float:
    """
    Calcula as Pot Odds (porcentagem mínima de vitória exigida pelo pote para pagar o call).
    Fórmula: Pot Odds (%) = [bet_to_call / (pot_size + bet_to_call)] * 100
    """
    if bet_to_call <= 0:
        return 0.0
    total_pot_after_call = pot_size + bet_to_call
    if total_pot_after_call <= 0:
        return 0.0
    return (bet_to_call / total_pot_after_call) * 100.0


def calculate_ev(
    p_win: float,
    p_lose: float,
    p_tie: float,
    pot_size: float,
    bet_to_call: float
) -> float:
    """
    Calcula o Valor Esperado (EV) da decisão de pagar (Call) no Texas Hold'em.
    Fórmula: EV = (P_win * V_pot) - (P_lose * V_bet) + [P_tie * (V_pot / 2)]
    """
    is_percentage = (p_win + p_lose + p_tie) > 1.5 or p_win > 1.0 or p_lose > 1.0
    pw = p_win / 100.0 if is_percentage else p_win
    pl = p_lose / 100.0 if is_percentage else p_lose
    pt = p_tie / 100.0 if is_percentage else p_tie

    return (pw * pot_size) - (pl * bet_to_call) + (pt * (pot_size / 2.0))


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
    draw_name: str = ""
) -> Dict[str, Any]:
    """
    Motor de Decisão Estratégica baseada em Teoria dos Jogos e Análise de Risco.
    """
    pot_odds = calculate_pot_odds(pot_size, bet_to_call)
    ev = calculate_ev(p_win, p_lose, p_tie, pot_size, bet_to_call)

    is_percentage = (p_win + p_lose + p_tie) > 1.5 or p_win > 1.0 or p_lose > 1.0
    pw = p_win / 100.0 if is_percentage else p_win
    pt = p_tie / 100.0 if is_percentage else p_tie
    equity = (pw + (pt / 2.0)) * 100.0

    action = "FOLD"
    recommended_amount = 0.0
    reason = ""

    # === FILTRO DE RANGE (DESCONTO DE EQUITY POST-FLOP) ===
    # Corrige o excesso de otimismo do Monte Carlo contra apostas adversárias
    bet_ratio_pot = bet_to_call / pot_size if pot_size > 0 else 0.0
    if state in ["FLOP", "TURN", "RIVER"] and bet_ratio_pot > 0.15:
        if hero_cards and board_cards:
            score = evaluate_7_cards(hero_cards + board_cards)
            # Se o Hero tem apenas High Card (score[0] == 0) e não tem outs formidáveis
            if score[0] == 0 and outs < 8:
                equity = equity * 0.50 # Reduz à metade
                ev = calculate_ev(equity, 100 - equity, 0.0, pot_size, bet_to_call)

    # 1. Blindagem de Sobrevivência contra All-Ins / Apostas Pesadas (>40% do stack)
    if bet_to_call > HEAVY_BET_STACK_RATIO * hero_stack and hero_stack > 0 and hero_cards:
        if state == "PRE_FLOP":
            tier = get_preflop_tier(hero_cards)
            if tier > 2:
                return {
                    "action": "FOLD",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": 0.0,
                    "reason": "Aposta pesada/All-in (>40% do stack). FOLD Mão insuficiente contra range polarizado."
                }
        else:
            if board_cards:
                score = evaluate_7_cards(hero_cards + board_cards)
                if score[0] < 2:
                    return {
                        "action": "FOLD",
                        "ev": ev,
                        "pot_odds": pot_odds,
                        "equity": equity,
                        "bet_to_call": bet_to_call,
                        "pot_size": pot_size,
                        "recommended_amount": 0.0,
                        "reason": "Aposta pesada/All-in (>40% do stack). FOLD exigiria no mínimo Dois Pares+."
                    }

    # 2. Regra PRE_FLOP Estratégica (Aggressividade Premium e Folds Trash)
    if state == "PRE_FLOP" and hero_cards:
        bet_ratio = bet_to_call / hero_stack if hero_stack > 0 else 1.0
        tier = get_preflop_tier(hero_cards)
        
        # Agressividade Premium (Tier 1: AA, KK, QQ, JJ, AKs, AKo)
        if tier == 1:
            if bet_to_call > 0 and bet_ratio < 0.25:
                # 3-Bet / Raise de Valor
                raise_amount = min(hero_stack, round(bet_to_call * 3.0, 1))
                return {
                    "action": "RAISE",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": raise_amount,
                    "reason": "Mão Premium (Tier 1). RAISE/3-BET obrigatório para extrair valor e isolar oponentes."
                }
            elif bet_to_call == 0:
                # Open Raise
                raise_amount = min(hero_stack, max(1.0, round(pot_size * 0.75, 1)))
                return {
                    "action": "BET",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": raise_amount,
                    "reason": "Mão Premium (Tier 1). OPEN RAISE de valor."
                }

        is_spec = is_speculative_hand(hero_cards)
        if bet_to_call > 0 and bet_ratio <= PREFLOP_SPECULATIVE_BET_RATIO and is_spec:
            return {
                "action": "CALL",
                "ev": ev,
                "pot_odds": pot_odds,
                "equity": equity,
                "bet_to_call": bet_to_call,
                "pot_size": pot_size,
                "recommended_amount": bet_to_call,
                "reason": "Aposta barata (<2.5% do stack) com mão suited/potencial. CALL por Implied Odds."
            }

        if tier == 4:
            if bet_to_call > 0:
                return {
                    "action": "FOLD",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": 0.0,
                    "reason": "Mão lixo (Tier 4) fora de posição contra aposta. FOLD imediato."
                }
            else:
                return {
                    "action": "CHECK",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": 0.0,
                    "reason": "Mão lixo (Tier 4) com Check gratuito."
                }

    # 3. Regra de Projetos e Semi-Blefe (Flop / Turn)
    if outs >= 8 and state in ["FLOP", "TURN"]:
        if bet_to_call == 0:
            action = "BET"
            recommended_amount = min(hero_stack, max(1.0, round(pot_size * 0.5, 1)))
            reason = f"Semi-Blefe com forte potencial ({draw_name} - {outs} Outs). Apostar 50% do pote."
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
        elif ev >= 0:
            action = "RAISE"
            recommended_amount = min(hero_stack, round(bet_to_call * 2.5, 1))
            reason = f"RAISE Semi-Blefe com {draw_name} ({outs} Outs). O EV já é positivo, aumentamos a fold equity."
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

    # 4. Regra de Micro-Aposta e Extração de Valor TPTK+ (Flop / Turn / River)
    if state in ["FLOP", "TURN", "RIVER"] and bet_to_call > 0:
        bet_ratio = bet_to_call / hero_stack if hero_stack > 0 else 1.0
        
        # Identificação de Força da Mão
        is_strong_hand = False
        has_draw = outs >= 4
        has_pair = False
        has_overcards = False
        
        if hero_cards and board_cards:
            score = evaluate_7_cards(hero_cards + board_cards)
            
            # Checa Top Pair Top Kicker (TPTK) ou melhor (Two Pair, Set, etc)
            # score[0] = Categoria da mão (1 = Par, 2 = Dois Pares...)
            if score[0] >= 2:
                is_strong_hand = True
            elif score[0] == 1:
                # É um par. Vamos ver se é Top Pair
                val_map = {
                    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
                    "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
                }
                try:
                    b_vals = [val_map[c[:-1].upper()] for c in board_cards if len(c) >= 2]
                    h_vals = [val_map[c[:-1].upper()] for c in hero_cards if len(c) >= 2]
                    max_board = max(b_vals) if b_vals else 0
                    
                    # Checa se o Par é com a carta mais alta do board (Top Pair)
                    is_top_pair = any(v == max_board for v in h_vals)
                    # Checa se a outra carta do Hero é K ou A (Top Kicker)
                    has_top_kicker = any(v >= 13 for v in h_vals)
                    
                    if is_top_pair and has_top_kicker:
                        is_strong_hand = True
                        
                    has_pair = True
                    if h_vals and b_vals and min(h_vals) > max_board:
                        has_overcards = True
                except Exception:
                    pass

        if bet_ratio <= MICRO_BET_STACK_RATIO:
            if is_strong_hand:
                # Oponente fez Micro-Aposta e Hero tem TPTK+
                raise_amount = min(hero_stack, round(bet_to_call * 3.5, 1))
                return {
                    "action": "RAISE",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": raise_amount,
                    "reason": "Micro-Aposta (<2% stack) detectada. Hero tem TPTK ou melhor. RAISE POR VALOR E PROTEÇÃO."
                }
            elif has_draw or has_pair or has_overcards:
                return {
                    "action": "CALL",
                    "ev": ev,
                    "pot_odds": pot_odds,
                    "equity": equity,
                    "bet_to_call": bet_to_call,
                    "pot_size": pot_size,
                    "recommended_amount": bet_to_call,
                    "reason": "Micro-Aposta (<2% stack) com par, overcards ou projeto (Implied Odds). CALL."
                }

    # 5. Regra Geral 1: Mesa em Check (bet_to_call == 0)
    if bet_to_call == 0:
        if equity > 55.0 and hero_stack > 0:
            action = "BET"
            bet_fraction = 0.66 if equity >= 70.0 else 0.50
            bet_val = max(1.0, round(pot_size * bet_fraction, 1)) if pot_size > 0 else 10.0
            recommended_amount = min(hero_stack, bet_val)
            pct_label = "66%" if equity >= 70.0 else "50%"
            reason = (
                f"Mesa em check e Equity favorável ({equity:.1f}% > 55%). "
                f"Apostar por valor (Value Bet de {pct_label} do pote: {recommended_amount:.1f} fichas)."
            )
        else:
            action = "CHECK"
            recommended_amount = 0.0
            reason = f"Aposta a pagar é 0 e Equity moderada ({equity:.1f}% <= 55%). Check gratuito para ver o próximo estágio."

    # 6. Regra Geral 2: Aposta ativa na mesa (bet_to_call > 0)
    else:
        if ev < 0 or equity < pot_odds:
            action = "FOLD"
            recommended_amount = 0.0
            if ev < 0 and equity < pot_odds:
                reason = (
                    f"EV negativo ({ev:+.2f} fichas) e Equity ({equity:.1f}%) inferior "
                    f"às Pot Odds exigidas ({pot_odds:.1f}%). Pagar é prejuízo a longo prazo."
                )
            elif ev < 0:
                reason = (
                    f"EV negativo ({ev:+.2f} fichas). Pagar aposta resulta em perda esperada a longo prazo."
                )
            else:
                reason = (
                    f"Equity insuficiente ({equity:.1f}%) frente às Pot Odds exigidas ({pot_odds:.1f}%). "
                    f"Fold recomendado para preservar stack."
                )
        else:
            if equity >= 60.0 and ev > 0:
                min_raise = bet_to_call * 2.0
                if hero_stack <= min_raise or hero_stack <= bet_to_call * 2.5:
                    action = "ALL-IN"
                    recommended_amount = hero_stack
                    reason = (
                        f"Equity dominante ({equity:.1f}% >= 60%) com stack curto/moderado ({hero_stack} fichas). "
                        f"Empurrar All-in para maximizar EV ({ev:+.2f} fichas)."
                    )
                else:
                    action = "RAISE"
                    recommended_amount = min(hero_stack, round(bet_to_call * 2.5, 1))
                    reason = (
                        f"Equity dominante ({equity:.1f}% >= 60%) e EV positivo ({ev:+.2f} fichas). "
                        f"Aumentar para extrair valor máximo do oponente."
                    )
            else:
                action = "CALL"
                recommended_amount = min(hero_stack, bet_to_call)
                reason = (
                    f"EV positivo ({ev:+.2f} fichas) e Equity ({equity:.1f}%) superior "
                    f"às Pot Odds ({pot_odds:.1f}%). Pagar a aposta é matematicamente lucrativo."
                )

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


def print_scenario_report(scenario_title: str, hand_desc: str, decision: Dict[str, Any]):
    """Imprime um relatório formatado e visual da decisão calculada."""
    print("=" * 75)
    print(f"ANÁLISE DE RISCO: {scenario_title}")
    print(f"Situação: {hand_desc}")
    print("-" * 75)
    print(f"  • Tamanho do Pote:       {decision['pot_size']:8.1f} fichas")
    print(f"  • Aposta a Pagar (Call): {decision['bet_to_call']:8.1f} fichas")
    print(f"  • Pot Odds Exigidas:     {decision['pot_odds']:8.2f}%")
    print(f"  • Equity do Hero:        {decision['equity']:8.2f}%")
    print(f"  • Valor Esperado (EV):   {decision['ev']:+8.2f} fichas")
    print("-" * 75)
    print(f"  >>> DECISÃO RECOMENDADA: [{decision['action']}] <<<")
    if decision["recommended_amount"] > 0:
        print(f"  • Valor Recomendado:     {decision['recommended_amount']:.1f} fichas")
    print(f"  • Justificativa:         {decision['reason']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("MOTOR DE ANÁLISE DE RISCO E VALOR ESPERADO (EV) - Seção 5 PAE")
    print("#" * 75 + "\n")

    hero_a = ["Ah", "Kh"]
    board_a = ["8h", "Qh", "Js", "2s", "3h"]
    pot_a = 85.0
    bet_a = 30.0
    hero_stack_a = 250.0

    try:
        from src.engine.evaluator import calculate_equity
        p_win_a, p_tie_a, p_lose_a, _, _ = calculate_equity(hero_a, board_a, num_opponents=1, iterations=10000)
    except Exception:
        p_win_a, p_tie_a, p_lose_a = 100.0, 0.0, 0.0

    decision_a = make_decision(
        p_win=p_win_a,
        p_lose=p_lose_a,
        p_tie=p_tie_a,
        pot_size=pot_a,
        bet_to_call=bet_a,
        hero_stack=hero_stack_a
    )

    print_scenario_report(
        scenario_title="CENÁRIO A (Hero no River com Nut Flush de Ás)",
        hand_desc=f"Hero={hero_a}, Board={board_a} | Pote={pot_a} fichas, Vilão aposta {bet_a} fichas",
        decision=decision_a
    )
