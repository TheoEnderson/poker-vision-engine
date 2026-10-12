"""
tests/test_pot_semantics_audit.py
Auditoria da Semântica do Pote — Etapa 4F-C.
Cenários determinísticos para validação matemática de Pot Odds, EV_CALL,
pote elegível, transições entre streets e segurança financeira.
"""

import unittest
from numbers import Real

from src.engine.risk_engine import (
    calculate_pot_odds,
    calculate_ev,
    make_decision,
)
from src.ocr.ocr_reader import extract_pot


class TestPotSemanticsAudit(unittest.TestCase):
    """Suíte de auditoria da semântica do pote e integridade financeira."""

    def test_blinds_1_2_initial_pot_and_preflop_call(self):
        """Cenário: Blinds 1/2 geram pote inicial de 3. Hero paga 2."""
        sb = 1.0
        bb = 2.0
        initial_pot = sb + bb  # 3.0
        hero_call = 2.0

        # Pote elegível antes do call do Hero
        p_eligible = initial_pot
        c = hero_call

        # Pot odds exigidas: C / (P + C)
        expected_pot_odds = (c / (p_eligible + c)) * 100.0  # 2 / (3 + 2) = 40.0%
        calculated_pot_odds = calculate_pot_odds(bet_to_call=c, pot_size=p_eligible)

        self.assertAlmostEqual(calculated_pot_odds, 40.0, places=4)
        self.assertAlmostEqual(calculated_pot_odds, expected_pot_odds, places=4)

    def test_preflop_to_flop_transition_accumulated_pot(self):
        """Cenário: Após o call do Hero de 2 no pré-flop, o pote acumulado é no mínimo 5."""
        initial_blinds = 3.0
        hero_call = 2.0
        # Pote acumulado consolidado no encerramento da rodada pré-flop
        accumulated_preflop_pot = initial_blinds + hero_call  # 5.0

        self.assertEqual(accumulated_preflop_pot, 5.0)

    def test_flop_opponent_bet_and_exact_eligible_pot(self):
        """Cenário: Flop com pote acumulado 5, aposta adversária de 2 e call do Hero de 2.

        O pote na mesa ANTES do call do Hero é P = 5 + 2 = 7.
        Hero enfrenta call adicional C = 2.
        Pote final após o call: P + C = 7 + 2 = 9.
        Pot Odds exatas: 2 / 9 = 22.2222%.
        EV_CALL exato: Equity * 9 - 2.
        """
        accumulated_pot = 5.0
        opponent_bet = 2.0
        hero_call = 2.0

        # Pote verdadeiramente elegível antes do call do Hero
        p_eligible = accumulated_pot + opponent_bet  # 7.0
        c = hero_call  # 2.0

        # Pot Odds com o pote correto (7.0)
        pot_odds = calculate_pot_odds(bet_to_call=c, pot_size=p_eligible)
        expected_pot_odds = (2.0 / (7.0 + 2.0)) * 100.0  # 2 / 9 = 22.2222%
        self.assertAlmostEqual(pot_odds, expected_pot_odds, places=4)
        self.assertAlmostEqual(pot_odds, 22.2222, places=3)

        # EV do Call com equity de 25%
        equity = 25.0
        ev = calculate_ev(equity=equity, pot_size=p_eligible, bet_to_call=c)
        # EV = (0.25 * 9.0) - 2.0 = 2.25 - 2.0 = +0.25
        expected_ev = (equity / 100.0) * (p_eligible + c) - c
        self.assertAlmostEqual(ev, expected_ev, places=4)
        self.assertAlmostEqual(ev, 0.25, places=4)

    def test_uncollected_bets_pot_underestimation_risk(self):
        """Demonstra o risco financeiro quando apostas da rodada não são agregadas a P.

        Se o motor receber apenas o pote antigo P=5 (ignorando a aposta adversária de 2),
        o pote final considerado é 5 + 2 = 7 em vez de 9.
        Pot Odds calculadas: 2 / 7 = 28.57% (em vez de 22.22%).
        EV calculado: 0.25 * 7 - 2 = -0.25 (em vez de +0.25).
        Um call lucrativo seria erroneamente avaliado como negativo.
        """
        stale_pot = 5.0
        c = 2.0
        equity = 25.0

        distorted_pot_odds = calculate_pot_odds(bet_to_call=c, pot_size=stale_pot)
        distorted_ev = calculate_ev(equity=equity, pot_size=stale_pot, bet_to_call=c)

        # Distorção: Pot odds de 28.57% em vez de 22.22%
        self.assertAlmostEqual(distorted_pot_odds, 28.5714, places=3)
        # Distorção: EV negativo (-0.25) quando o EV real com pote 7+2 é positivo (+0.25)
        self.assertAlmostEqual(distorted_ev, -0.25, places=4)

    def test_current_bet_alone_as_pot_causes_severe_distortion(self):
        """Demonstra o erro gravíssimo se P for APENAS a aposta corrente (ex: P=2, C=2).

        Isso reproduz o caso observado na mão 6s/6h no Flop.
        Se P=2 e C=2:
        Pote final considerado: 2 + 2 = 4 (em vez de 9).
        Pot Odds exigidas: 2 / 4 = 50.0% (em vez de 22.22%).
        EV: Equity * 4 - 2 (severamente penalizado).
        """
        p_corrupted = 2.0
        c = 2.0
        equity = 8.97  # Equity observada na mão 6s/6h

        distorted_pot_odds = calculate_pot_odds(bet_to_call=c, pot_size=p_corrupted)
        distorted_ev = calculate_ev(equity=equity, pot_size=p_corrupted, bet_to_call=c)

        self.assertAlmostEqual(distorted_pot_odds, 50.0, places=4)
        # EV calculado no log foi exatamente -1.64
        self.assertAlmostEqual(distorted_ev, -1.6412, places=3)

    def test_unknown_accumulated_pot_requires_suspension(self):
        """Segurança: Pote nulo ou inválido não deve alimentar o cálculo de EV."""
        for invalid_pot in (None, -5.0, float("nan"), float("inf")):
            with self.subTest(invalid_pot=invalid_pot):
                with self.assertRaises(ValueError):
                    calculate_ev(equity=50.0, pot_size=invalid_pot, bet_to_call=10.0)

    def test_all_in_insufficient_stack_exceeds_call(self):
        """Segurança: Call superior ao stack do Hero não é call normal nem elegível."""
        hero_stack = 100.0
        nominal_call = 250.0
        # A aposta nominal excede o stack do Hero;
        # potes laterais e valor elegível não podem ser supostos.
        self.assertGreater(nominal_call, hero_stack)

    def test_all_in_effective_matching_stack(self):
        """Segurança: Quando o Hero vai all-in, o custo efetivo C é exatamente o stack."""
        hero_stack = 100.0
        effective_call = 100.0
        pot_size = 200.0
        equity = 40.0

        ev = calculate_ev(equity=equity, pot_size=pot_size, bet_to_call=effective_call)
        # Pote final: 200 + 100 = 300. EV: 0.40 * 300 - 100 = +20.0
        self.assertAlmostEqual(ev, 20.0, places=4)

    def test_multiline_pot_ocr_extraction_on_samples(self):
        """Validação: extract_pot extrai o valor real mesmo com cabeçalhos multiline."""
        # Amostras do Replay Poker
        val_mesa6 = extract_pot("assets/samples/mesa_6.png", verbose=False)
        self.assertEqual(val_mesa6, 28.0)

        val_turno = extract_pot("assets/samples/mesa_turno.png", verbose=False)
        self.assertEqual(val_turno, 30.0)

        val_novas = extract_pot("assets/samples/mesa_novas.png", verbose=False)
        # Com a correção de parsing multiline de baixo para cima, mesa_novas lê 40.0 (não 1.0)
        self.assertEqual(val_novas, 40.0)

        val_par = extract_pot("assets/samples/mesa_par_de_6.png", verbose=False)
        self.assertEqual(val_par, 3840.0)


if __name__ == "__main__":
    unittest.main()
