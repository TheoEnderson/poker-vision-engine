"""Ranges reais integrados à FSM e ao cache, sem OCR ou escrita de dataset."""

import io
import unittest
from contextlib import redirect_stdout

from src.engine.evaluator import FULL_DECK
from src.engine.range_model import OpponentModel
from src.engine.state_machine import HandState
from tests import test_app_cache
from tests.test_range_model import FLOP, TURN, RIVER, HERO


class TestRangeIntegration(unittest.TestCase):
    def setUp(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = list(HERO)
        self.model = OpponentModel(FULL_DECK)
        self.app.opponent_model = self.model

    def process(self, board=FLOP, pot=318, call=10):
        with redirect_stdout(io.StringIO()):
            return self.app.process_cards(board, pot, call, False)

    def test_pot_changes_recalculate_analysis_without_reweighting_ranges(self):
        self.process()
        first = dict(self.model.weights)
        for pot in [320, 318, 295, 400, 318]:
            self.process(pot=pot)
            self.assertEqual(self.model.weights, first)
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.make_decision.call_count, 6)
        self.assertIs(self.calculate_equity.call_args.kwargs["opp_range_percent"], self.model)

    def test_preflop_pot_changes_do_not_infer_new_opponent_actions(self):
        self.process([], 25)
        first = dict(self.model.weights)
        for pot in [30, 35, 100, 25]:
            self.process([], pot)
            self.assertEqual(self.model.weights, first)

    def test_same_capture_reuses_cache_and_weights(self):
        first = self.process()
        weights = dict(self.model.weights)
        self.assertIs(self.process(), first)
        self.assertEqual(self.model.weights, weights)
        self.calculate_equity.assert_called_once()

    def test_stack_opponents_and_position_invalidate_analysis_not_ranges(self):
        self.process()
        weights = dict(self.model.weights)
        for attr, value in [("hero_stack", 900), ("num_opponents", 3), ("hero_position", "CO")]:
            setattr(self.app, attr, value)
            self.process()
            self.assertEqual(self.model.weights, weights)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.make_decision.call_count, 4)

    def test_new_streets_accumulate_each_evidence_once(self):
        expected = OpponentModel(FULL_DECK)
        for board, state in [(FLOP, HandState.FLOP), (TURN, HandState.TURN), (RIVER, HandState.RIVER)]:
            self.process(board)
            expected.apply_postflop_board(board, 318)
            self.assertEqual(self.app.state_machine.current_state, state)
            self.assertEqual(self.model.weights, expected.weights)
            self.process(board, 320)
            self.assertEqual(self.model.weights, expected.weights)

    def test_confirmed_new_hand_resets_ranges_and_evidence_history(self):
        self.process(RIVER)
        first = dict(self.model.weights)
        with redirect_stdout(io.StringIO()):
            self.app.reset_hand(hero_cards=HERO, reason="Nova mão confirmada no teste")
        self.assertEqual(self.model.weights, OpponentModel(FULL_DECK).weights)
        self.assertIsNone(self.app.last_analysis)
        self.process(RIVER)
        self.assertEqual(self.model.weights, first)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_rejected_cards_and_finances_never_change_ranges(self):
        previous = self.process()
        weights = dict(self.model.weights)
        for board, pot in [([], 320), (["Desconhecida"] * 5, 320),
                           (FLOP + ["8h"], 320), (TURN, None)]:
            result = self.process(board, pot)
            self.assertFalse(result["observation_valid"])
            self.assertEqual(self.model.weights, weights)
            self.assertIs(self.app.last_analysis, previous)
        self.calculate_equity.assert_called_once()
        self.process(TURN)
        self.assertNotEqual(self.model.weights, weights)

    def test_unknown_call_does_not_consume_first_board_evidence(self):
        prior = dict(self.model.weights)
        self.process(call=None)
        self.assertEqual(self.model.weights, prior)
        self.calculate_equity.assert_not_called()
        self.process()
        expected = OpponentModel(FULL_DECK)
        expected.apply_postflop_board(FLOP, 318)
        self.assertEqual(self.model.weights, expected.weights)

    def test_unknown_call_then_recovery_never_reapplies_existing_evidence(self):
        for board in ([], FLOP):
            with self.subTest(board=board):
                self.app.reset_hand(hero_cards=HERO, reason="Cenário independente")
                self.calculate_equity.reset_mock()
                self.make_decision.reset_mock()
                self.process(board)
                weights = dict(self.model.weights)
                revision = self.model.revision
                for pot in (318, 320, 318):
                    pending = self.process(board, pot, call=None)
                    self.assertEqual(pending["action"], "AGUARDANDO VALOR DO CALL")
                    self.assertFalse(pending["recalculated"])
                    self.assertEqual(self.model.weights, weights)
                    self.assertEqual(self.model.revision, revision)
                self.calculate_equity.assert_called_once()
                recovered = self.process(board)
                self.assertEqual(self.model.weights, weights)
                self.assertEqual(self.model.revision, revision)
                self.assertEqual(recovered["range_revision"], revision)
                self.calculate_equity.assert_called_once()
                self.assertEqual(self.make_decision.call_count, 2)
                self.assertIs(self.process(board), recovered)
                self.calculate_equity.assert_called_once()
                self.assertEqual(self.make_decision.call_count, 2)

    def test_unknown_call_on_new_street_defers_evidence_until_recovery(self):
        self.process()
        weights = dict(self.model.weights)
        revision = self.model.revision
        self.process(TURN, call=None)
        self.assertEqual(self.model.weights, weights)
        self.assertEqual(self.model.revision, revision)
        self.calculate_equity.assert_called_once()
        self.process(TURN)
        expected = OpponentModel(FULL_DECK)
        expected.apply_postflop_board(FLOP, 318)
        expected.apply_postflop_board(TURN, 318)
        self.assertEqual(self.model.weights, expected.weights)
        self.assertEqual(self.model.revision, revision + 1)
        self.process(TURN, 320)
        self.assertEqual(self.model.weights, expected.weights)
        self.assertEqual(self.model.revision, revision + 1)

    def test_range_reset_invalidates_cached_analysis(self):
        first = self.process()
        self.model.reset()
        second = self.process()
        self.assertIsNot(first, second)
        self.assertEqual(self.calculate_equity.call_count, 2)


if __name__ == "__main__":
    unittest.main()
