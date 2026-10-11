"""Sequências de mãos no --live; captura/OCR e matemática são fronteiras controladas."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import run_live
from src.engine.state_machine import HandState
from tests import test_app_cache


HERO = ["5h", "9h"]
NEXT_HERO = ["Ac", "8c"]
FLOP = ["Ks", "10d", "5d"]
TURN = FLOP + ["6d"]
RIVER = TURN + ["2d"]
NEXT_FLOP = ["4s", "3h", "2c"]


def frame(hero=HERO, board=RIVER, pot=100, turn=True, stack=1000, position="BTN"):
    return dict(hero=list(hero), board=list(board), pot=pot, turn=turn,
                stack=stack, position=position)


def complete_hand():
    return [frame(board=[], pot=30), frame(board=FLOP, pot=50),
            frame(board=TURN, pot=80), frame()]


class TestLiveHands(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp

    def live(self, frames, debug=False):
        self.app.hero_cards = list(HERO)
        self.app.auto_detect_hero = True
        output = io.StringIO()
        def turn_info(f, **kw):
            return dict(is_hero_turn=f["turn"], action_type="CALL" if f["turn"] else "WAITING",
                        bet_to_call=10, available_actions=["FOLD", "CALL"] if f["turn"] else [])
        with patch("main.PokerAnalyticsApp", return_value=self.app), \
                patch("main.grab_screen", side_effect=frames + [KeyboardInterrupt]), \
                patch("main.find_and_detect_hero_cards", side_effect=lambda f, **kw: (f["hero"], None)), \
                patch("main.detect_board", side_effect=lambda f, **kw: f["board"]), \
                patch("main.extract_pot", side_effect=lambda f, **kw: f["pot"]), \
                patch("main.detect_turn_and_bet_to_call", side_effect=turn_info), \
                patch("main.extract_hero_stack", side_effect=lambda f, **kw: f["stack"]), \
                patch("main.extract_hero_position", side_effect=lambda f, **kw: f["position"]), \
                patch("main.detect_active_opponents", return_value=2), \
                patch("main.time.sleep"), \
                patch.object(self.app, "render_dashboard", wraps=self.app.render_dashboard) as dashboard, \
                redirect_stdout(output):
            run_live(debug_hands=debug)
        self.analyses = [call.args[0] for call in dashboard.call_args_list]
        return output.getvalue()

    def assert_new_hand(self, hero, board):
        self.assertEqual(self.app.hero_cards, hero)
        self.assertEqual(self.app.state_machine.board, board)
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.last_analysis["hero_cards"], hero)
        self.assertEqual(self.app.last_analysis["board"], board)
        self.app.opponent_model.reset.assert_called_once()

    def test_complete_hand_then_another_without_large_pot_drop(self):
        next_frame = frame(NEXT_HERO, [], pot=80)
        self.live(complete_hand() + [next_frame, next_frame])
        self.assert_new_hand(NEXT_HERO, [])
        self.assertEqual(self.app.state_machine.current_state, HandState.PRE_FLOP)
        self.assertEqual(self.calculate_equity.call_count, 5)

    def test_river_end_and_redeal_while_not_hero_turn(self):
        self.live([frame(), frame([], [], pot=0, turn=False),
                   frame(NEXT_HERO, [], pot=80, turn=False),
                   frame(NEXT_HERO, [], pot=80, turn=False),
                   frame(NEXT_HERO, [], pot=80)])
        self.assert_new_hand(NEXT_HERO, [])
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_new_hand_after_hero_win_with_larger_pot_and_updated_context(self):
        new = frame(NEXT_HERO, [], pot=150, stack=1500, position="CO")
        self.live(complete_hand() + [frame([], [], pot=0, turn=False), new, new])
        self.assert_new_hand(NEXT_HERO, [])
        self.assertEqual(self.app.last_analysis["hero_stack"], 1500)
        self.assertEqual(self.app.last_analysis["hero_position"], "CO")

    def test_new_hand_after_fold_before_flop(self):
        new = frame(NEXT_HERO, [], pot=40)
        self.live([frame(board=[], pot=30), frame([], [], pot=0, turn=False), new, new])
        self.assert_new_hand(NEXT_HERO, [])

    def test_new_hand_after_explicit_showdown(self):
        self.app.hero_cards = list(HERO)
        self.app.process_cards(RIVER, 100, 10, False)
        self.app.state_machine.transition_to(HandState.SHOWDOWN)
        new = frame(NEXT_HERO, NEXT_FLOP, pot=100)
        self.live([new, new])
        self.assert_new_hand(NEXT_HERO, NEXT_FLOP)

    def test_identical_hole_cards_with_observed_end(self):
        new = frame(HERO, [], pot=80)
        self.live(complete_hand() + [frame([], [], pot=0, turn=False), new, new])
        self.assert_new_hand(HERO, [])
        self.assertEqual(self.calculate_equity.call_count, 5)

    def test_identical_hole_cards_with_larger_pot(self):
        new = frame(HERO, [], pot=150)
        self.live([frame(), frame([], [], pot=0, turn=False), new, new])
        self.assert_new_hand(HERO, [])

    def test_identical_hole_cards_new_flop_without_intermediate_frames(self):
        new = frame(HERO, NEXT_FLOP, pot=150)
        self.live([frame(), new, new])
        self.assert_new_hand(HERO, NEXT_FLOP)

    def test_new_hand_with_larger_pot_and_no_observed_end(self):
        new = frame(NEXT_HERO, [], pot=150)
        self.live([frame(), new, new])
        self.assert_new_hand(NEXT_HERO, [])

    def test_new_hand_directly_on_flop_after_missing_preflop(self):
        new = frame(NEXT_HERO, NEXT_FLOP, pot=150)
        self.live(complete_hand() + [new, new])
        self.assert_new_hand(NEXT_HERO, NEXT_FLOP)
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)

    def test_river_ocr_failure_then_recovery_is_not_new_hand(self):
        self.live([frame(), frame(board=["Desconhecida"] * 5), frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.state_machine.board, RIVER)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.app.opponent_model.reset.assert_not_called()

    def test_temporary_hero_loss_then_recovery_does_not_reset(self):
        self.live([frame(), frame([], RIVER, turn=False), frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.hero_cards, HERO)
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_single_conflicting_capture_cannot_reset(self):
        output = self.live([frame(), frame(NEXT_HERO, NEXT_FLOP, pot=150), frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(len(self.analyses), 2)
        self.assertIn("CAPTURA INCERTA", output)

    def test_large_pot_drop_does_not_bypass_corroboration(self):
        self.live([frame(), frame(NEXT_HERO, [], pot=10), frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_new_hand_does_not_require_any_pot_change(self):
        new = frame(NEXT_HERO, [], pot=100)
        self.live([frame(), new, new])
        self.assert_new_hand(NEXT_HERO, [])

    def test_empty_board_alone_never_confirms_new_hand(self):
        empty = frame(board=[], pot=80)
        self.live([frame(), empty, empty, frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_several_hands_without_restarting_program(self):
        new = frame(NEXT_HERO, NEXT_FLOP, pot=150)
        third = frame(["Qh", "Qc"], [], pot=200)
        self.live(complete_hand() + [new, new, frame(NEXT_HERO, NEXT_FLOP + ["7d"], 180),
                                   frame(NEXT_HERO, NEXT_FLOP + ["7d", "Js"], 200), third, third])
        self.assertEqual(self.app.hero_cards, ["Qh", "Qc"])
        self.assertEqual(self.app.state_machine.current_state, HandState.PRE_FLOP)
        self.assertEqual(self.app.state_machine.hand_generation, 2)
        self.assertEqual(self.app.opponent_model.reset.call_count, 2)

    def test_actual_range_weights_are_reset_not_carried_to_next_hand(self):
        from src.engine.evaluator import FULL_DECK
        from src.engine.range_model import OpponentModel
        model = OpponentModel(FULL_DECK)
        self.app.opponent_model = model
        new = frame(NEXT_HERO, NEXT_FLOP, pot=150)
        with patch.object(model, "reset", wraps=model.reset) as reset:
            self.live(complete_hand() + [new, new])
        reset.assert_called_once()
        expected = OpponentModel(FULL_DECK)
        expected.apply_postflop_board(NEXT_FLOP, 150)
        self.assertEqual(model.weights, expected.weights)
        self.assertIsNot(self.analyses[-1], self.analyses[-2])
        self.assertEqual(self.analyses[-1]["hero_cards"], NEXT_HERO)
        self.assertEqual(self.analyses[-2]["hero_cards"], HERO)

    def test_identical_preflop_data_after_gap_stays_uncertain_without_evidence(self):
        current = frame(board=[], pot=30)
        output = self.live([current, frame([], [], pot=0, turn=False), current, current])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(len(self.analyses), 1)
        self.assertIn("CAPTURA INCERTA", output)

    def test_alternating_candidates_do_not_count_as_corroboration(self):
        self.live([frame(), frame(NEXT_HERO, NEXT_FLOP, pot=150),
                   frame(["Qh", "Qc"], [], pot=150),
                   frame(NEXT_HERO, NEXT_FLOP, pot=150), frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_changed_hero_on_same_old_board_is_not_new_hand(self):
        suspicious = frame(NEXT_HERO, RIVER, pot=150)
        self.live([frame(), suspicious, suspicious, frame()])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.calculate_equity.call_count, 1)

    def test_diagnostics_are_optional_and_explain_rejected_reset(self):
        output = self.live([frame(), frame(NEXT_HERO, NEXT_FLOP, pot=150)], debug=True)
        self.assertIn("[DEBUG MÃO]", output)
        self.assertIn("estado_anterior", output)
        self.assertIn("hero_observado", output)
        self.assertIn("board_observado", output)
        self.assertIn("criterios", output)
        self.assertIn("aguardando observação compatível", output)

    def test_normal_mode_has_no_diagnostic_dump(self):
        output = self.live([frame()])
        self.assertNotIn("[DEBUG MÃO]", output)


if __name__ == "__main__":
    unittest.main()
