"""Cartas fisicamente únicas antes da FSM, do cache e da matemática."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.engine.state_machine import HandState, PokerHandStateMachine
from tests import test_app_cache, test_live_hands


HERO = ["Jd", "2s"]
TURN = ["5s", "Kc", "Jc", "7c"]
VALID_RIVER = TURN + ["Qd"]
INVALID_RIVER = TURN + ["Jd"]


class TestCardUniqueness(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp

    def process(self, board, show=False):
        with redirect_stdout(io.StringIO()):
            return self.app.process_cards(board, 120, 20, show)

    def assert_rejected_without_math(self, hero, board):
        self.app.hero_cards = hero
        history = list(self.app.state_machine.history)
        result = self.process(board)
        self.assertFalse(result.get("observation_valid", True))
        self.assertFalse(result["action_available"])
        self.assertFalse(result["recalculated"])
        self.assertIn("duplicad", result["reason"].lower())
        self.assertEqual(self.app.state_machine.current_state, HandState.WAITING_HAND)
        self.assertEqual(self.app.state_machine.board, [])
        self.assertEqual(self.app.state_machine.history, history)
        self.assertIsNone(self.app.last_analysis)
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()
        self.app.opponent_model.assert_not_called()
        self.assertEqual(self.app.opponent_model.mock_calls, [])
        self.assertEqual(self.app.dataset_logger.mock_calls, [])

    def test_duplicate_hero_cards(self):
        self.assert_rejected_without_math(["Jd", "Jd"], [])

    def test_duplicate_board_cards(self):
        self.assert_rejected_without_math(HERO, ["5s", "Kc", "5s"])

    def test_card_shared_by_hero_and_board(self):
        self.assert_rejected_without_math(HERO, INVALID_RIVER)

    def test_duplicate_identity_is_case_insensitive(self):
        self.assert_rejected_without_math(["Jd", "JD"], [])

    def test_ten_aliases_are_the_same_physical_card(self):
        self.assert_rejected_without_math(["10d", "2s"], ["5s", "Kc", "Td"])

    def test_board_aliases_are_rejected_by_fsm_itself(self):
        fsm = PokerHandStateMachine()
        with redirect_stdout(io.StringIO()):
            self.assertFalse(fsm.update_board(["10d", "Td", "5s"], has_hero_cards=True))
        self.assertEqual(fsm.current_state, HandState.WAITING_HAND)

    def test_valid_pair_with_different_suits_is_accepted(self):
        self.app.hero_cards = ["Jd", "Jh"]
        result = self.process(TURN)
        self.assertEqual(result["board"], TURN)
        self.assertEqual(result["hero_cards"], ["Jd", "Jh"])
        self.calculate_equity.assert_called_once()

    def test_repeated_empty_slots_are_not_duplicate_cards(self):
        self.app.hero_cards = HERO
        result = self.process(["Vazio"] * 5)
        self.assertEqual(result["state"], "PRE_FLOP")
        self.calculate_equity.assert_called_once()

    def test_invalid_river_preserves_turn_history_cache_and_ranges(self):
        self.app.hero_cards = HERO
        previous = self.process(TURN)
        history = list(self.app.state_machine.history)
        range_calls = list(self.app.opponent_model.mock_calls)
        result = self.process(INVALID_RIVER)
        self.assertFalse(result["observation_valid"])
        self.assertFalse(result["action_available"])
        self.assertEqual(result["state"], "TURN")
        self.assertEqual(result["board"], TURN)
        self.assertEqual(result["board_observation"], INVALID_RIVER)
        self.assertEqual(self.app.state_machine.history, history)
        self.assertIs(self.app.last_analysis, previous)
        self.assertEqual(self.app.opponent_model.mock_calls, range_calls)
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.app.total_recalculations, 1)

    def test_recovery_accepts_valid_river_and_recalculates(self):
        self.app.hero_cards = HERO
        self.process(TURN)
        self.process(INVALID_RIVER)
        result = self.process(VALID_RIVER)
        self.assertEqual(result["state"], "RIVER")
        self.assertEqual(result["board"], VALID_RIVER)
        self.assertIsNot(result.get("observation_valid"), False)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.app.state_machine.hand_generation, 0)

    def test_recovery_of_same_valid_capture_can_reuse_cache(self):
        self.app.hero_cards = HERO
        previous = self.process(TURN)
        self.process(INVALID_RIVER)
        self.assertIs(self.process(TURN), previous)
        self.calculate_equity.assert_called_once()

    def test_dashboard_never_presents_old_decision_as_current(self):
        self.app.hero_cards = HERO
        self.process(TURN)
        output = io.StringIO()
        with redirect_stdout(output):
            self.app.process_cards(INVALID_RIVER, 120, 20, True)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output.getvalue())
        self.assertIn("duplicad", output.getvalue().lower())
        self.assertNotIn("ANÁLISE INFORMATIVA", output.getvalue())
        self.assertNotIn("RECOMENDAÇÃO PARA A CAPTURA:", output.getvalue())

    def test_auto_detected_duplicate_hero_recovers_on_next_static_frame(self):
        self.app.hero_cards = []
        self.app.auto_detect_hero = True
        with patch("main.find_and_detect_hero_cards", side_effect=[
                (["Jd", "Jd"], None), (HERO, None)]), \
                patch("main.extract_hero_position", return_value="BTN"), \
                patch("main.detect_board", return_value=[]), \
                redirect_stdout(io.StringIO()):
            invalid = self.app.process_frame(None, 120, 20, False)
            self.assertFalse(invalid["observation_valid"])
            self.calculate_equity.assert_not_called()
            self.assertEqual(self.app.state_machine.current_state, HandState.WAITING_HAND)
            valid = self.app.process_frame(None, 120, 20, False)
        self.assertEqual(valid["hero_cards"], HERO)
        self.assertEqual(valid["state"], "PRE_FLOP")
        self.calculate_equity.assert_called_once()

    def test_duplicate_capture_cannot_confirm_new_hand(self):
        self.app.hero_cards = HERO
        self.process(VALID_RIVER)
        previous = self.app.last_analysis
        history = list(self.app.state_machine.history)
        for _ in range(3):
            status = self.app.observe_hand(["Ac", "8c"], ["4s", "3h", "Ac"], 80)
            self.assertEqual(status, "UNCERTAIN")
            self.assertIn("duplicad", self.app.last_hand_observation["motivo"].lower())
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.state_machine.history, history)
        self.assertIs(self.app.last_analysis, previous)
        self.app.opponent_model.reset.assert_not_called()

    def test_duplicate_capture_breaks_new_hand_corroboration(self):
        self.app.hero_cards = HERO
        self.process(VALID_RIVER)
        next_hero = ["Ac", "8c"]
        next_board = ["4s", "3h", "2c"]
        self.assertEqual(self.app.observe_hand(next_hero, next_board, 80), "UNCERTAIN")
        self.assertEqual(self.app.observe_hand(next_hero, ["4s", "3h", "Ac"], 80), "UNCERTAIN")
        self.assertEqual(self.app.observe_hand(next_hero, next_board, 80), "UNCERTAIN")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.app.observe_hand(next_hero, next_board, 80), "NEW_HAND")
        self.assertEqual(self.app.hero_cards, next_hero)
        self.assertEqual(self.app.state_machine.board, next_board)


class TestLiveCardUniqueness(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp
    live = test_live_hands.TestLiveHands.live

    def test_live_duplicate_is_suspended_then_recovers_without_reset(self):
        frame = test_live_hands.frame
        valid = test_live_hands.RIVER
        invalid = valid[:-1] + [test_live_hands.HERO[0]]
        output = self.live([frame(), frame(board=invalid), frame()])
        self.assertIn("CAPTURA INCERTA", output)
        self.assertIn("DUPLICAD", output.upper())
        self.assertEqual(len(self.analyses), 2)
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.app.state_machine.board, valid)
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.app.opponent_model.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
