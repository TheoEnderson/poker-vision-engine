"""
tests/test_dealer_divergence_and_continuity.py
Suíte de testes para a Etapa 4F-D:
Validação de divergência do dealer, continuidade da mão, rejeição de falsos positivos,
rotação de posições e integridade das recomendações do PAE.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock
import numpy as np

from main import run_live, PokerAnalyticsApp
from src.engine.evaluator import FULL_DECK
from src.engine.range_model import OpponentModel
from src.engine.risk_engine import make_decision
from src.engine.state_machine import HandState
from src.ocr.ocr_reader import extract_hero_position
from tests import test_app_cache
from tests.test_same_street_decisions import (
    HERO, OTHER, FLOP, TURN, RIVER, observation, context
)


def capture(*args, position="BTN", **kwargs):
    return dict(observation(*args, **kwargs), position=position)


class TestDealerDivergenceAndContinuity(unittest.TestCase):
    """Cenários determinísticos para validação da FSM e detecção de Dealer/Posição."""

    def setUp(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = list(HERO)
        self.app.hero_position = "Desconhecida"
        self.model = OpponentModel(FULL_DECK)
        self.app.opponent_model = self.model

    def live(self, frames, showdown=False, debug=False):
        output = io.StringIO()
        snapshots = []

        def grab():
            if not frames:
                raise KeyboardInterrupt
            if showdown and len(frames) == 2:
                self.app.state_machine.transition_to(HandState.SHOWDOWN)
            f = frames.pop(0)
            snapshots.append(dict(
                generation=self.app.state_machine.hand_generation,
                weights=dict(self.model.weights),
                revision=self.model.revision,
                cache=self.app.last_analysis,
                equity_key=getattr(self.app, "_equity_cache_key", None)
            ))
            return f

        with patch("main.PokerAnalyticsApp", return_value=self.app), \
             patch("main.grab_screen", side_effect=grab), \
             patch("main.find_and_detect_hero_cards", side_effect=lambda f, **kw: (f["hero"], None)), \
             patch("main.detect_board", side_effect=lambda f, **kw: f["board"]), \
             patch("main.extract_pot", side_effect=lambda f, **kw: f["pot"]), \
             patch("main.detect_turn_and_bet_to_call", side_effect=lambda f, **kw: context(f)), \
             patch("main.extract_hero_position", side_effect=lambda f, **kw: f["position"]), \
             patch("main.extract_hero_stack", side_effect=lambda f, **kw: f["stack"]), \
             patch("main.detect_active_opponents", side_effect=lambda f, **kw: f.get("opponents", 2)), \
             patch("main.time.sleep"), \
             patch("main.make_decision", wraps=make_decision) as risk, \
             patch("src.engine.blueprint.PreflopBlueprint.sample_action",
                   side_effect=lambda strategy: max(strategy, key=strategy.get)), \
             patch.object(self.model, "reset", wraps=self.model.reset) as reset, \
             patch.object(self.app, "render_dashboard", wraps=self.app.render_dashboard) as dashboard, \
             redirect_stdout(output):
            run_live(debug_hands=debug)

        self.snapshots = snapshots
        self.risk = risk
        self.reset_count = reset.call_count
        self.analyses = [call.args[0] for call in dashboard.call_args_list]
        return output.getvalue()

    # 1. Hero realmente em BTN
    def test_hero_really_on_btn(self):
        """Hero no assento Mid Right e Dealer no Mid Right confirmam posição BTN."""
        self.live([capture(position="BTN")])
        self.assertEqual(self.app.hero_position, "BTN")
        self.assertEqual(len(self.analyses), 1)
        self.assertEqual(self.analyses[0]["hero_position"], "BTN")

    # 2. Hero em CO
    def test_hero_on_co(self):
        """Hero no assento Mid Right e Dealer no Bot Right confirmam posição CO."""
        self.live([capture(position="CO")])
        self.assertEqual(self.app.hero_position, "CO")
        self.assertEqual(len(self.analyses), 1)
        self.assertEqual(self.analyses[0]["hero_position"], "CO")

    # 3. Dealer rotacionando entre mãos
    def test_dealer_rotating_between_hands(self):
        """Dealer muda entre mãos sucessivas: posição atualizada e não congelada."""
        # Mão 1: BTN
        # Intervalo observado (mesa limpa, hero fora)
        # Mão 2: SB
        hand2 = capture(hero=OTHER, pot=15, position="SB")
        self.live([
            capture(position="BTN"),
            capture(hero=[], pot=0, turn=False, position="BTN"),
            hand2, hand2
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.hero_position, "SB")
        self.assertEqual(self.analyses[-1]["hero_position"], "SB")

    # 4. Falso positivo de dealer rejeitado
    def test_false_positive_dealer_rejected(self):
        """Textura de feltro plano sem disco brilhante é rejeitada pelo detector."""
        fake_felt = np.full((1080, 1920, 3), [73, 140, 101], dtype=np.uint8)
        diag = {}
        pos = extract_hero_position(fake_felt, diagnostics=diag)
        self.assertEqual(pos, "Desconhecida")
        self.assertIn("abaixo do limiar", diag.get("reason", ""))

    # 5. Divergência temporária no flop
    def test_temporary_divergence_on_flop(self):
        """Flop com 1 frame de leitura divergente de dealer atualiza o board sem travar."""
        self.live([
            capture(position="BTN"),
            capture(board=FLOP, position="CO"),  # divergência temporária no flop
            capture(board=FLOP, position="BTN")  # recuperação imediata
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.state_machine.board, list(FLOP))
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.app.hero_position, "BTN")

    # 6. Mão estável com OCR do dealer oscilando
    def test_stable_hand_with_oscillating_dealer_ocr(self):
        """Durante mão ativa, oscilações no OCR do dealer preservam a posição confirmada."""
        self.live([
            capture(position="BTN"),
            capture(position="Desconhecida"),  # dealer temporariamente não lido
            capture(position="BTN")
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.hero_position, "BTN")

    # 7. Cartas do Hero temporariamente ilegíveis
    def test_hero_cards_temporarily_unreadable(self):
        """Queda de 1 frame nas cartas do Hero é protegida pelo Iron Lock."""
        self.live([
            capture(pot=10, call=0),
            capture(hero=[], pot=10, call=0),  # falha efêmera de OCR das cartas
            capture(pot=10, call=0)
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.hero_cards, list(HERO))

    # 8. Nova mão com mesmo par de cartas
    def test_new_hand_with_same_hole_cards(self):
        """Hero recebe mesmas cartas, mas encerramento e novo dealer confirmam nova mão."""
        hand2 = capture(hero=HERO, pot=15, position="CO")
        self.live([
            capture(position="BTN"),
            capture(hero=[], pot=0, turn=False, position="BTN"),  # intervalo observado
            hand2, hand2
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.reset_count, 1)
        self.assertEqual(self.app.hero_position, "CO")

    # 9. Nova mão após fold
    def test_new_hand_after_fold(self):
        """Hero folda pré-flop, mesa esvazia e nova mão é recebida."""
        hand2 = capture(hero=OTHER, pot=15, position="CO")
        self.live([
            capture(pot=20, call=5),
            capture(hero=[], pot=20, turn=False),  # fold
            capture(hero=[], pot=0, turn=False),   # mesa limpa
            hand2, hand2
        ])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.hero_cards, list(OTHER))

    # 10. Recuperação de uma captura incerta
    def test_recovery_from_uncertain_capture(self):
        """Captura com board incerto suspende momentaneamente, recuperando no frame seguinte."""
        self.live([
            capture(board=FLOP),
            capture(board=["Kd", "Desconhecida", "Qh"]),  # board incerto
            capture(board=FLOP)                           # recuperação
        ])
        self.assertEqual(self.app.state_machine.board, list(FLOP))
        self.assertEqual(len(self.analyses), 2)

    # 11. Alteração de pote e call durante suspensão
    def test_pot_and_call_change_during_suspension(self):
        """Valores financeiros alterados durante suspensão são recalculados após recuperação."""
        self.live([
            capture(pot=20, call=5),
            capture(board=["Desconhecida"]),          # suspenso
            capture(pot=60, call=25, stack=975)       # recuperado com novo pote/call
        ])
        self.assertEqual(len(self.analyses), 2)
        last = self.analyses[-1]
        self.assertEqual(last["pot_size"], 60)
        self.assertEqual(last["bet_to_call"], 25)
        # Pot odds = 25 / (60 + 25) = 29.41%
        self.assertAlmostEqual(last["pot_odds"], 25 / 85 * 100, places=2)

    # 12. Ausência de reutilização de decisão antiga
    def test_absence_of_stale_decision_reuse(self):
        """Quando o valor da aposta a pagar aumenta, a decisão não reutiliza cache antigo."""
        self.live([
            capture(pot=20, call=5),
            capture(pot=50, call=30, stack=970)
        ])
        self.assertEqual(len(self.analyses), 2)
        self.assertNotEqual(self.analyses[0]["bet_to_call"], self.analyses[1]["bet_to_call"])
        self.assertFalse(self.analyses[1].get("cached", False))


if __name__ == "__main__":
    unittest.main()
