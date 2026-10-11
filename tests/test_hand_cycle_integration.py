"""4F-A: observações integradas, sem presumir que uma recomendação foi executada.

Captura/OCR, sorteio e escrita são fronteiras controladas. O ciclo --live,
FSM, ranges, validação financeira, EV, blueprint e CFR+ são reais.
Posição é a saída existente do reconhecedor de dealer/nome, não um ID de mão.
"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import run_live
from src.engine.evaluator import FULL_DECK
from src.engine.range_model import OpponentModel
from src.engine.risk_engine import make_decision
from src.engine.state_machine import HandState
from tests import test_app_cache
from tests.test_same_street_decisions import HERO, OTHER, FLOP, TURN, RIVER, observation, context


def capture(*args, position="BTN", **kwargs):
    return dict(observation(*args, **kwargs), position=position)


class TestHandCycleIntegration(unittest.TestCase):
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
            # Estado ANTES de cada captura: detecta reset/alteração prematura.
            snapshots.append(dict(generation=self.app.state_machine.hand_generation,
                                  weights=dict(self.model.weights), revision=self.model.revision,
                                  cache=self.app.last_analysis, equity_key=getattr(self.app, "_equity_cache_key", None)))
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

    def assert_continuity(self, board=(), simulations=1):
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.reset_count, 0)
        self.assertEqual(self.app.state_machine.board, list(board))
        self.assertEqual(self.calculate_equity.call_count, simulations)

    def assert_redeal(self, hero, board=()):
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.reset_count, 1)
        self.assertEqual(self.app.hero_cards, list(hero))
        self.assertEqual(self.app.state_machine.board, list(board))
        self.assertFalse(self.analyses[-1]["equity_cached"])

    def test_check_raise_and_repeated_decisions_recalculate_only_decision(self):
        self.live([capture(), capture(turn=False), capture(pot=25, call=10),
                   capture(turn=False), capture(pot=45, call=20, stack=990)])
        self.assert_continuity()
        self.assertEqual(self.risk.call_count, 3)
        self.assertEqual([a["action"] for a in self.analyses], ["CHECK", "FOLD", "FOLD"])
        for analysis, ev in zip(self.analyses, (2.05, 4.35, 6.65)):
            self.assertAlmostEqual(analysis["ev"], ev)
        self.assertEqual(self.snapshots[1]["weights"], self.model.weights)

    def test_call_and_new_raise_do_not_end_street(self):
        hero = ["Kh", "Qh"]
        self.app.hero_cards = hero
        self.live([capture(hero=hero, pot=25, call=10), capture(hero=hero, turn=False),
                   capture(hero=hero, pot=65, call=20, stack=990)])
        self.assert_continuity()
        self.assertEqual([a["action"] for a in self.analyses], ["CALL", "CALL"])
        self.assertEqual([a["recommended_amount"] for a in self.analyses], [10, 20])

    def test_full_hand_and_multiple_postflop_decisions(self):
        frames = [capture()]
        for board in (FLOP, TURN, RIVER):
            frames += [capture(board=board, pot=100), capture(board=board, turn=False),
                       capture(board=board, pot=140, call=20)]
        self.live(frames)
        self.assert_continuity(RIVER, simulations=4)
        self.assertEqual(self.risk.call_count, 7)
        self.assertEqual([a["state"] for a in self.analyses],
                         ["PRE_FLOP", "FLOP", "FLOP", "TURN", "TURN", "RIVER", "RIVER"])

    def test_fold_preflop_then_new_cards_without_zero_uses_dealer_change(self):
        new = capture(hero=OTHER, pot=15, position="CO")
        self.live([capture(pot=25, call=10), capture(hero=[], pot=25, turn=False), new, new])
        self.assert_redeal(OTHER)
        self.assertEqual(self.app.hero_position, "CO")
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_fold_preflop_then_same_cards_with_blinds_and_dealer_change(self):
        new = capture(pot=15, position="CO")
        self.live([capture(pot=25, call=10), capture(hero=[], pot=25, turn=False), new, new])
        self.assert_redeal(HERO)
        expected = OpponentModel(FULL_DECK)
        expected.apply_action_heuristic(15, "PRE_FLOP")
        self.assertEqual(self.model.weights, expected.weights)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_quick_new_hand_without_any_gap_requires_cards_and_dealer(self):
        new = capture(hero=OTHER, pot=15, position="CO")
        self.live([capture(), new, new])
        self.assert_redeal(OTHER)

    def test_same_cards_and_new_dealer_without_gap_is_ambiguous(self):
        new = capture(pot=15, position="CO")
        output = self.live([capture(), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)

    def test_known_zero_end_still_confirms_without_position(self):
        new = capture(pot=15, position="Desconhecida")
        self.live([capture(position="Desconhecida"),
                   capture(hero=[], pot=0, turn=False, position="Desconhecida"), new, new])
        self.assert_redeal(HERO)

    def test_different_cards_after_gap_with_no_zero_or_position_stay_uncertain(self):
        unknown = "Desconhecida"
        new = capture(hero=OTHER, pot=15, position=unknown)
        output = self.live([capture(position=unknown),
                            capture(hero=[], pot=25, turn=False, position=unknown), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)

    def test_same_cards_after_gap_with_no_zero_or_position_stay_uncertain(self):
        unknown = "Desconhecida"
        new = capture(pot=15, position=unknown)
        output = self.live([capture(position=unknown),
                            capture(hero=[], pot=25, turn=False, position=unknown), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)

    def test_one_and_two_divergent_card_reads_are_not_a_redeal(self):
        wrong = capture(hero=["Kd", "4d"], pot=25, call=10)
        self.live([capture(), wrong, wrong, capture(pot=25, call=10)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.app.hero_cards, HERO)

    def test_single_dealer_and_card_error_recovers_without_reset(self):
        self.live([capture(), capture(hero=OTHER, position="CO"), capture()])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)

    def test_alternating_dealer_reads_do_not_confirm_candidate(self):
        self.live([capture(), capture(hero=OTHER, position="CO"),
                   capture(hero=OTHER, position="MP"), capture()])
        self.assert_continuity()

    def test_unreadable_dealer_does_not_confirm_candidate(self):
        self.live([capture(), capture(hero=OTHER, position="CO"),
                   capture(hero=OTHER, position="Desconhecida"), capture()])
        self.assert_continuity()

    def test_same_cards_after_river_without_zero_or_pot_change(self):
        new = capture(pot=100, position="CO")
        self.live([capture(board=RIVER, pot=100), capture(hero=[], pot=100, turn=False), new, new])
        self.assert_redeal(HERO)

    def test_showdown_and_new_hand_without_zero(self):
        new = capture(hero=OTHER, pot=15, position="CO")
        self.live([capture(board=RIVER, pot=100), new, new], showdown=True)
        self.assert_redeal(OTHER)

    def test_showdown_same_cards_without_redistribution_evidence_stays_uncertain(self):
        new = capture(pot=100)
        output = self.live([capture(board=RIVER, pot=100), new, new], showdown=True)
        # SHOWDOWN confirma fim, não nova distribuição: só o board desapareceu.
        self.assert_continuity(RIVER)
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)

    def test_new_hand_can_be_captured_directly_on_flop_without_zero(self):
        new = capture(hero=OTHER, board=FLOP, pot=15, position="CO")
        self.live([capture(), new, new])
        self.assert_redeal(OTHER, FLOP)

    def test_current_position_recovery_resolves_gap_without_reset(self):
        unknown = capture(pot=25, call=10, position="Desconhecida")
        self.live([capture(), capture(hero=[], turn=False), unknown, unknown,
                   capture(pot=25, call=10)])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 2)
        self.assertTrue(self.analyses[-1]["action_available"])
        self.assertTrue(self.analyses[-1]["equity_cached"])

    def test_pot_drop_and_different_cards_without_dealer_evidence_are_ambiguous(self):
        new = capture(hero=OTHER, pot=15)
        self.live([capture(pot=100, call=10), capture(hero=[], pot=100, turn=False), new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_dealer_error_alone_recovers_on_same_postflop_board(self):
        self.live([capture(board=FLOP), capture(board=FLOP, position="CO"), capture(board=FLOP)])
        self.assert_continuity(FLOP)
        self.assertEqual(len(self.analyses), 2)

    def test_contradictory_capture_breaks_dealer_candidate_corroboration(self):
        new = capture(hero=OTHER, position="CO")
        # Dados ausentes não refutam candidato; uma carta impossível refuta.
        invalid = dict(new, board=[OTHER[0], "2d", "3s"])
        self.live([capture(), new, invalid, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_unknown_buttons_do_not_corroborate_same_card_redistribution(self):
        # UNKNOWN não equivale a WAITING: há botões, mas sua ação é ilegível.
        initial = capture()
        unknown = capture(hero=[], turn=False)
        new = capture(position="CO")
        original_context = context

        def read_context(f):
            result = original_context(f)
            if f is unknown:
                result["action_type"] = "UNKNOWN"
            return result

        with patch(f"{__name__}.context", side_effect=read_context):
            self.live([initial, unknown, new, new])
        self.assert_continuity()
        self.assertEqual(len(self.analyses), 1)

    def test_caches_and_ranges_change_only_after_confirmation(self):
        new = capture(hero=OTHER, pot=5, position="CO")
        self.live([capture(pot=100, call=10), capture(hero=[], pot=100, turn=False), new, new])
        self.assert_redeal(OTHER)
        before_gap, before_candidate, before_confirmation = self.snapshots[1:]
        for snapshot in (before_candidate, before_confirmation):
            self.assertIs(snapshot["cache"], before_gap["cache"])
            self.assertEqual(snapshot["equity_key"], before_gap["equity_key"])
            self.assertEqual(snapshot["revision"], before_gap["revision"])
            self.assertEqual(snapshot["weights"], before_gap["weights"])
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertNotEqual(self.model.weights, before_gap["weights"])

    def test_rejected_finances_after_confirmation_do_not_produce_recommendation(self):
        new = capture(hero=OTHER, pot=25, call=20, stack=5, position="CO")
        good = dict(new, stack=1000)
        self.live([capture(), new, new, good])
        self.assert_redeal(OTHER)
        self.assertFalse(self.analyses[1]["observation_valid"])
        self.assertFalse(self.analyses[1]["action_available"])
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_several_preflop_hands_without_zero_or_restart(self):
        second = capture(hero=OTHER, pot=15, position="CO")
        third = capture(hero=["Kh", "Qh"], pot=15, position="MP")
        self.live([capture(), second, second, third, third])
        self.assertEqual(self.app.state_machine.hand_generation, 2)
        self.assertEqual(self.reset_count, 2)
        self.assertEqual(self.calculate_equity.call_count, 3)
        self.assertEqual(self.app.hero_cards, ["Kh", "Qh"])


if __name__ == "__main__":
    unittest.main()
