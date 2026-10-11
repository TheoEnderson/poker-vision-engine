"""Regressões de street, validade da captura e identidade da mão, sem OCR real."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import run_live
from src.engine.state_machine import HandState, PokerHandStateMachine
from tests import test_app_cache


FLOP = ["8s", "4c", "Jd"]
TURN = FLOP + ["2c"]
# Ah já pertence ao Hero da fixture compartilhada.
RIVER = TURN + ["Qh"]


class TestBoardState(unittest.TestCase):
    def setUp(self):
        self.fsm = PokerHandStateMachine()

    def update(self, cards):
        with redirect_stdout(io.StringIO()):
            return self.fsm.update_board(cards, has_hero_cards=True)

    def test_normal_streets(self):
        for board, state in [([], HandState.PRE_FLOP), (FLOP, HandState.FLOP),
                             (TURN, HandState.TURN), (RIVER, HandState.RIVER)]:
            self.assertTrue(self.update(board))
            self.assertEqual(self.fsm.current_state, state)
            self.assertEqual(self.fsm.board, board)

    def test_flop_then_empty_read_preserves_board_and_history(self):
        self.update(FLOP)
        history = list(self.fsm.history)
        self.assertFalse(self.update([]))
        self.assertEqual(self.fsm.current_state, HandState.FLOP)
        self.assertEqual(self.fsm.board, FLOP)
        self.assertEqual(self.fsm.history, history)
        self.assertEqual(self.fsm.board_read_status, "UNRECOGNIZED")

    def test_turn_then_partial_read_does_not_regress_to_flop(self):
        self.update(TURN)
        self.assertFalse(self.update(FLOP + ["Desconhecida", "Vazio"]))
        self.assertEqual(self.fsm.current_state, HandState.TURN)
        self.assertEqual(self.fsm.board, TURN)
        self.assertEqual(self.fsm.board_read_status, "PARTIAL")

    def test_river_then_total_recognition_failure(self):
        self.update(RIVER)
        self.assertFalse(self.update(["Desconhecida"] * 5))
        self.assertEqual(self.fsm.current_state, HandState.RIVER)
        self.assertEqual(self.fsm.board, RIVER)

    def test_empty_slots_are_not_proof_of_new_hand(self):
        self.update(FLOP)
        self.assertFalse(self.update(["Vazio"] * 5))
        self.assertEqual(self.fsm.current_state, HandState.FLOP)

    def test_unknown_extra_card_is_not_a_fully_recognized_flop(self):
        self.update(FLOP)
        self.assertFalse(self.update(FLOP + ["Desconhecida", "Vazio"]))
        self.assertEqual(self.fsm.board, FLOP)

    def test_unknown_board_at_start_does_not_confirm_preflop(self):
        self.assertFalse(self.update(["Desconhecida", "Vazio"]))
        self.assertEqual(self.fsm.current_state, HandState.WAITING_HAND)

    def test_empty_unused_slots_allow_normal_flop(self):
        self.update([])
        self.assertTrue(self.update(FLOP + ["Vazio", "Vazio"]))
        self.assertEqual(self.fsm.board, FLOP)

    def test_conflicting_recognized_cards_do_not_replace_known_board(self):
        self.update(FLOP)
        self.assertFalse(self.update(["9s", "4c", "Jd", "2c"]))
        self.assertEqual(self.fsm.board, FLOP)

    def test_recovery_after_invalid_read(self):
        self.update(FLOP)
        self.update([])
        self.assertTrue(self.update(TURN))
        self.assertEqual(self.fsm.current_state, HandState.TURN)

    def test_explicit_reset_allows_next_preflop(self):
        self.update(RIVER)
        with redirect_stdout(io.StringIO()):
            self.fsm.reset_hand()
        self.assertEqual(self.fsm.current_state, HandState.WAITING_HAND)
        self.assertEqual(self.fsm.board, [])
        self.assertEqual(self.fsm.history[-1]["reason"], "Reset de mão")
        self.assertEqual(self.fsm.board_read_status, "RESET")
        self.assertEqual(self.fsm.hand_generation, 1)
        self.assertTrue(self.update([]))
        self.assertEqual(self.fsm.current_state, HandState.PRE_FLOP)

    def test_street_regression_requires_explicit_reset(self):
        for state in (HandState.FLOP, HandState.TURN, HandState.RIVER):
            with self.subTest(state=state):
                self.assertFalse(self.fsm._is_transition_allowed(state, HandState.PRE_FLOP, 0))

    def test_initial_sync_with_known_hero_is_preserved(self):
        self.fsm = PokerHandStateMachine(allow_initial_sync=False)
        self.assertTrue(self.update(TURN))
        self.assertEqual(self.fsm.current_state, HandState.TURN)


class TestHandLifecycle(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp

    def analyze(self, board, pot=318):
        return self.app.process_cards(board, pot, 10, False, action_context={
            "is_hero_turn": True, "action_type": "CALL", "bet_to_call": 10,
            "available_actions": ["FOLD", "CALL", "RAISE"],
        })

    def test_new_hero_flag_is_consumed_on_first_analysis(self):
        self.app.auto_detect_hero = True
        self.app._new_hand_hero_cards_set = True
        self.analyze([])
        self.assertFalse(self.app._new_hand_hero_cards_set)

    def test_invalid_board_blocks_current_recommendation_and_calculation(self):
        first = self.analyze(FLOP)
        self.assertTrue(first["action_available"])
        weights_updates = self.app.opponent_model.apply_postflop_board.call_count
        invalid = self.analyze([], pot=320)
        self.assertFalse(invalid["action_available"])
        self.assertFalse(invalid["observation_valid"])
        self.assertEqual(invalid["state"], "FLOP")
        self.assertEqual(invalid["board"], FLOP)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.app.opponent_model.apply_postflop_board.call_count, weights_updates)
        output = io.StringIO()
        with redirect_stdout(output):
            self.app.render_dashboard(invalid)
        self.assertIn("CAPTURA INCERTA", output.getvalue())
        self.assertNotIn("RECOMENDAÇÃO PARA A CAPTURA:", output.getvalue())
        self.assertNotIn("RECÁLCULO EXECUTADO", output.getvalue())

    def test_valid_capture_recovers_cached_analysis_after_uncertainty(self):
        first = self.analyze(FLOP)
        self.analyze(["Desconhecida"] * 5)
        recovered = self.analyze(FLOP)
        self.assertIs(recovered, first)
        self.assertTrue(recovered["action_available"])
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_partial_turn_and_failed_river_never_revalidate_cached_action(self):
        for board, invalid in [(TURN, FLOP + ["Desconhecida"]),
                               (RIVER, ["Desconhecida"] * 5)]:
            with self.subTest(board=board):
                first = self.analyze(board)
                calls = self.calculate_equity.call_count
                second = self.analyze(invalid)
                self.assertFalse(second["action_available"])
                self.assertEqual(second["board"], board)
                self.assertIs(self.app.last_analysis, first)
                self.assertEqual(self.calculate_equity.call_count, calls)

    def test_initial_uncertainty_does_not_create_analysis_or_leave_flag_set(self):
        self.app._new_hand_hero_cards_set = True
        analysis = self.analyze(["Desconhecida", "Vazio"])
        self.assertFalse(analysis["observation_valid"])
        self.assertFalse(self.app._new_hand_hero_cards_set)
        self.assertIsNone(self.app.last_analysis)
        self.calculate_equity.assert_not_called()
        with redirect_stdout(io.StringIO()):
            self.app.render_dashboard(analysis)

    def test_confirmed_new_hand_with_identical_hero_invalidates_cache(self):
        old = self.analyze([])
        hero = list(self.app.hero_cards)
        self.app.reset_hand(hero_cards=hero, reason="Nova mão confirmada pelo chamador")
        new = self.analyze([])
        self.assertIsNot(new, old)
        self.assertEqual(new["hero_cards"], hero)
        self.assertEqual(new["state"], "PRE_FLOP")
        self.assertFalse(self.app._new_hand_hero_cards_set)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.app.opponent_model.reset.assert_called_once()

    def test_reset_clears_old_hero_even_if_old_flag_was_set(self):
        self.app.auto_detect_hero = True
        self.analyze(FLOP)
        self.app._new_hand_hero_cards_set = True
        self.app.reset_hand()
        self.assertEqual(self.app.hero_cards, [])
        self.assertFalse(self.app._new_hand_hero_cards_set)
        self.assertIsNone(self.app.last_analysis)
        self.assertEqual(self.app.last_board, [])
        self.assertEqual(self.app.last_state, HandState.WAITING_HAND)
        self.assertEqual(self.app.hero_position, "Desconhecida")

    def test_direct_machine_reset_does_not_reuse_old_hero_or_analysis(self):
        self.app.auto_detect_hero = True
        self.analyze(FLOP)
        self.app._new_hand_hero_cards_set = True
        self.app.state_machine.reset_hand()
        result = self.analyze([])
        self.assertEqual(self.app.hero_cards, [])
        self.assertFalse(result["action_available"])
        self.assertEqual(self.calculate_equity.call_count, 1)

    def live(self, frames):
        output = io.StringIO()
        self.app.auto_detect_hero = True
        with patch("main.PokerAnalyticsApp", return_value=self.app), \
                patch("main.grab_screen", side_effect=frames + [KeyboardInterrupt]), \
                patch("main.find_and_detect_hero_cards", side_effect=lambda f, **kw: (f["hero"], None)), \
                patch("main.detect_turn_and_bet_to_call", return_value={"is_hero_turn": True, "action_type": "CALL", "bet_to_call": 10, "available_actions": ["FOLD", "CALL"]}), \
                patch("main.detect_board", side_effect=lambda f, **kw: f["board"]), \
                patch("main.extract_pot", side_effect=lambda f, **kw: f["pot"]), \
                patch("main.detect_active_opponents", return_value=2), \
                patch("main.extract_hero_position", return_value="BTN"), \
                patch("main.extract_hero_stack", return_value=1000), \
                patch("main.time.sleep"), redirect_stdout(output):
            run_live()
        return output.getvalue()

    def test_live_pot_drop_and_empty_board_alone_do_not_confirm_new_hand(self):
        hero = list(self.app.hero_cards)
        output = self.live([{"hero": hero, "board": FLOP, "pot": 318},
                            {"hero": hero, "board": [], "pot": 30}])
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)
        self.assertEqual(self.app.hero_cards, hero)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertIn("CAPTURA INCERTA", output)
        self.assertIn("Ciclos Otimizados por Cache:       0", output)

    def test_live_confirmed_new_hand_can_have_identical_hero_cards(self):
        hero = list(self.app.hero_cards)
        output = self.live([{"hero": hero, "board": FLOP, "pot": 318},
                            {"hero": [], "board": [], "pot": 30},
                            {"hero": hero, "board": ["Vazio"] * 5, "pot": 30},
                            {"hero": hero, "board": ["Vazio"] * 5, "pot": 30}])
        self.assertEqual(self.app.state_machine.current_state, HandState.PRE_FLOP)
        self.assertEqual(self.app.hero_cards, hero)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertTrue(any("Nova mão confirmada" in h["reason"] for h in self.app.state_machine.history))
        self.assertIn("Nova mão confirmada", output)

    def test_live_confirms_changed_hero_with_clear_board_and_pot_drop(self):
        hero = list(self.app.hero_cards)
        self.live([{"hero": hero, "board": FLOP, "pot": 318},
                   {"hero": ["4h", "9d"], "board": ["Vazio"] * 5, "pot": 30},
                   {"hero": ["4h", "9d"], "board": ["Vazio"] * 5, "pot": 30}])
        self.assertEqual(self.app.state_machine.current_state, HandState.PRE_FLOP)
        self.assertEqual(self.app.hero_cards, ["4h", "9d"])
        self.assertFalse(self.app._new_hand_hero_cards_set)

    def test_live_changed_hero_without_confirmation_never_uses_old_cards(self):
        hero = list(self.app.hero_cards)
        output = self.live([{"hero": hero, "board": FLOP, "pot": 318},
                            {"hero": ["4h", "9d"], "board": [], "pot": 320}])
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)
        self.assertIn("MUDANÇA DO HERO SEM NOVA MÃO CONFIRMADA", output)

    def test_live_unknown_board_is_not_clear_even_with_other_evidence(self):
        hero = list(self.app.hero_cards)
        self.live([{"hero": hero, "board": FLOP, "pot": 318},
                   {"hero": [], "board": [], "pot": 30},
                   {"hero": hero, "board": ["Desconhecida"] * 5, "pot": 30}])
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.app.state_machine.hand_generation, 0)

    def test_live_hero_absence_alone_does_not_reset_after_frame_threshold(self):
        hero = list(self.app.hero_cards)
        self.live([{"hero": hero, "board": FLOP, "pot": 318}]
                  + [{"hero": [], "board": [], "pot": 318}] * 9)
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)
        self.assertEqual(self.app.hero_cards, hero)
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)


if __name__ == "__main__":
    unittest.main()
