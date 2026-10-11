"""Confirmação financeira e limites de ações, sem OCR real nem arquivos."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from main import PokerAnalyticsApp, run_live
from src.engine.game_tree import GameNode, GameTreeBuilder
from src.engine.risk_engine import make_decision
from src.ocr import ocr_reader
from tests import test_app_cache, test_live_hands


HERO = ["10s", "Jh"]
TURN = ["Qs", "7s", "6s", "Jd"]
RIVER = TURN + ["2c"]


def call_context(amount):
    return dict(is_hero_turn=True, action_type="CALL", bet_to_call=amount,
                available_actions=["FOLD", "CALL", "RAISE"])


class TestFinancialContext(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp

    def process(self, stack, pot=215, call=20, board=TURN, context=None, show=False):
        self.app.hero_cards = list(HERO)
        self.app.hero_stack = stack
        self.make_decision.return_value.update(action="CALL", recommended_amount=call)
        with redirect_stdout(io.StringIO()):
            return self.app.process_cards(board, pot, call, show,
                                          action_context=call_context(call) if context is None else context)

    def assert_suspended(self, result):
        self.assertFalse(result.get("financial_valid", True))
        self.assertFalse(result["action_available"])
        self.assertFalse(result["recalculated"])
        self.assertIn("FINANCEIRO", result["action"])

    def test_stack_above_call_reaches_engine_unchanged(self):
        result = self.process(2415)
        self.assertTrue(result["action_available"])
        self.assertEqual(result["recommended_amount"], 20)
        self.assertEqual(self.make_decision.call_args.kwargs["hero_stack"], 2415)
        self.assertEqual(self.make_decision.call_args.kwargs["bet_to_call"], 20)

    def test_stack_equal_to_call_is_explicit_call_all_in(self):
        result = self.process(20)
        self.assertTrue(result["action_available"])
        self.assertEqual(result["action"], "CALL")
        self.assertEqual(result["recommended_amount"], 20)
        self.assertTrue(result.get("call_is_all_in", False))
        output = io.StringIO()
        with redirect_stdout(output):
            self.app.render_dashboard(result)
        self.assertIn("CALL ALL-IN", output.getvalue())

    def test_stack_below_nominal_call_is_not_a_normal_call(self):
        result = self.process(5)
        self.assert_suspended(result)
        self.assertIn("all-in", result["reason"].lower())
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()
        self.assertIsNone(self.app.last_analysis)
        self.assertEqual(self.app.opponent_model.mock_calls, [])

    def test_small_legitimate_stack_is_not_rejected_by_size(self):
        self.assertTrue(self.process(5, call=1)["action_available"])
        self.assertTrue(self.process(0.5, call=0.5)["action_available"])

    def test_effective_all_in_amount_observed_on_button_is_supported(self):
        # A aposta nominal dos rivais pode ser 20; o botão PAGO 5 informa
        # o desembolso efetivo. Não se inventa min(20, 5) no orquestrador.
        context = ocr_reader.parse_action_button_text("DESISTO PAGO 5")
        result = self.process(5, call=context["bet_to_call"], context=context)
        self.assertTrue(result["action_available"])
        self.assertTrue(result.get("call_is_all_in", False))
        self.assertEqual(self.make_decision.call_args.kwargs["bet_to_call"], 5)

    def test_zero_stack_has_no_current_action(self):
        self.assert_suspended(self.process(0, call=0))
        self.make_decision.assert_not_called()

    def test_nonfinite_negative_or_missing_values_never_reach_engine(self):
        for field in ("stack", "pot", "call"):
            values = [-1, float("nan"), float("inf")]
            if field != "call":
                values.append(None)
            for value in values:
                with self.subTest(field=field, value=value):
                    args = dict(stack=2415, pot=215, call=20)
                    args[field] = value
                    self.assert_suspended(self.process(**args))
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()

    def test_observed_stack_oscillation_does_not_pollute_cache_or_ranges(self):
        previous = self.process(2415)
        range_calls = list(self.app.opponent_model.mock_calls)
        invalid = self.process(5, pot=295)
        self.assert_suspended(invalid)
        self.assertIs(self.app.last_analysis, previous)
        self.assertEqual(self.app.opponent_model.mock_calls, range_calls)
        recovered = self.process(2375, pot=335, board=RIVER)
        self.assertTrue(recovered["action_available"])
        self.assertEqual(recovered["hero_stack"], 2375)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual([c.kwargs["hero_stack"] for c in self.make_decision.call_args_list], [2415, 2375])

    def test_recovery_of_identical_valid_context_reuses_cache(self):
        previous = self.process(2415)
        self.assert_suspended(self.process(5))
        self.assertIs(self.process(2415), previous)
        self.calculate_equity.assert_called_once()

    def test_missing_current_stack_cannot_validate_old_cached_recommendation(self):
        previous = self.process(2415)
        invalid = self.process(None)
        self.assert_suspended(invalid)
        self.assertIs(self.app.last_analysis, previous)
        self.assertIsNone(invalid["financial_observation"]["hero_stack"])
        self.calculate_equity.assert_called_once()

    def test_static_unreadable_pot_does_not_invent_default(self):
        previous = self.process(2415)
        self.app.auto_detect_pot = True
        with patch("main.extract_pot", return_value=None) as pot_ocr, \
                patch("main.extract_hero_position", return_value="BTN"), \
                patch("main.detect_board", return_value=TURN):
            invalid = self.app.process_frame(None, bet_to_call=20, show_dashboard=False)
        self.assert_suspended(invalid)
        self.assertIsNone(invalid["financial_observation"]["pot_size"])
        self.assertIs(self.app.last_analysis, previous)
        self.assertIsNone(pot_ocr.call_args.kwargs["default_value"])
        self.calculate_equity.assert_called_once()

    def test_repeated_unaffordable_call_does_not_become_confirmed(self):
        for _ in range(3):
            self.assert_suspended(self.process(5))
        self.make_decision.assert_not_called()

    def test_mismatched_button_amount_is_blocked_before_strategy(self):
        self.assert_suspended(self.process(2415, call=20, context=call_context(40)))
        self.make_decision.assert_not_called()

    def test_invalid_financial_capture_has_no_current_dashboard_recommendation(self):
        previous = self.process(2415)
        invalid = self.process(5)
        output = io.StringIO()
        with redirect_stdout(output):
            self.app.render_dashboard(invalid)
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output.getvalue())
        self.assertIn("5", output.getvalue())
        self.assertIn("20", output.getvalue())
        self.assertNotIn("RECOMENDAÇÃO PARA A CAPTURA:", output.getvalue())
        self.assertNotIn("ANÁLISE INFORMATIVA", output.getvalue())
        self.assertIs(self.app.last_analysis, previous)

    def test_action_gate_independently_rejects_amount_above_stack(self):
        result = dict(action="CALL", recommended_amount=20, bet_to_call=20,
                      hero_stack=5, recalculated=True)
        PokerAnalyticsApp._with_action_context(result, call_context(20))
        self.assertFalse(result["action_available"])
        self.assertIn("stack", result["action_status"].lower())

    def test_actual_risk_engine_receives_consistent_all_in_context(self):
        self.app.hero_cards = list(HERO)
        self.app.hero_stack = 20
        with patch("main.make_decision", wraps=make_decision) as risk, \
                patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"CALL": 1.0}), \
                redirect_stdout(io.StringIO()):
            result = self.app.process_cards(TURN, 215, 20, False, action_context=call_context(20))
        self.assertEqual(risk.call_args.kwargs["hero_stack"], 20)
        self.assertEqual(risk.call_args.kwargs["bet_to_call"], 20)
        self.assertEqual(result["recommended_amount"], 20)
        self.assertTrue(result["action_available"])
        self.assertTrue(result.get("call_is_all_in", False))

    def test_legacy_engine_and_tree_diverge_for_unaffordable_call(self):
        """Caracterização da limitação matemática não corrigida nesta etapa.

        Chamadas diretas ao motor ainda não têm a proteção do orquestrador.
        """
        node = GameNode(player=0, pot_size=295, hero_stack=5, villain_stack=5,
                        board_cards=TURN, current_street="TURN", bet_to_call=20)
        child = GameTreeBuilder()._apply_action(node, "CALL")
        with patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"CALL": 1.0}):
            decision = make_decision(p_win=40, p_tie=2, p_lose=58, pot_size=295,
                                     bet_to_call=20, hero_stack=5, state="TURN",
                                     hero_cards=HERO, board_cards=TURN)
        self.assertEqual(child.hero_stack, 0)
        self.assertEqual(child.pot_size, 300)  # A árvore paga somente 5.
        self.assertEqual(decision["recommended_amount"], 20)  # Saída legada não limita o call.


class TestStackOCRConfirmation(unittest.TestCase):
    def setUp(self):
        self.badge = np.zeros((60, 180, 3), dtype=np.uint8)

    def test_disagreement_between_preprocessing_modes_is_unknown(self):
        with patch("pytesseract.image_to_string", side_effect=["Hero\n5", "Hero\n2,415"]):
            self.assertIsNone(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True))

    def test_single_partial_result_is_not_confirmation(self):
        readings = iter(["Hero\n5"])
        with patch("pytesseract.image_to_string", side_effect=lambda *a, **kw: next(readings, "")):
            self.assertIsNone(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True))

    def test_split_numeric_tokens_are_not_reduced_to_last_digit(self):
        with patch("pytesseract.image_to_string", return_value="Hero\n241 5"):
            self.assertIsNone(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True))

    def test_confirmed_small_large_and_zero_stacks(self):
        for text, expected in [("5", 5), ("2,415", 2415), ("0", 0), ("12.50", 12.5),
                               ("2415.50", 2415.5), ("2.415,50", 2415.5), ("0.5", 0.5)]:
            with self.subTest(text=text), patch("pytesseract.image_to_string", return_value=f"Hero\n{text}"):
                self.assertEqual(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True), expected)

    def test_ocr_disagreement_recovers_on_next_read(self):
        with patch("pytesseract.image_to_string", side_effect=["5", "2415", "2375", "2375"]):
            self.assertIsNone(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True))
            self.assertEqual(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True), 2375)

    def test_negative_or_malformed_stack_is_not_a_positive_guess(self):
        for text in ("-5", "1.2.3", "24 15"):
            with self.subTest(text=text), patch("pytesseract.image_to_string", return_value=text):
                self.assertIsNone(ocr_reader.parse_badge_crop(self.badge, require_confirmation=True))

    def test_unreadable_pot_can_explicitly_remain_unknown(self):
        frame = np.zeros((800, 1200, 3), dtype=np.uint8)
        with patch("pytesseract.image_to_string", return_value="???"):
            self.assertIsNone(ocr_reader.extract_pot(frame, default_value=None, verbose=False))

    def test_strict_extraction_does_not_return_historical_stack(self):
        empty_data = dict(text=[], left=[], top=[], width=[], height=[])
        with patch.object(ocr_reader, "last_hero_name_coords", (10, 10, 100, 50)), \
                patch.object(ocr_reader, "parse_badge_crop", return_value=None), \
                patch("pytesseract.image_to_data", return_value=empty_data):
            value = ocr_reader.extract_hero_stack(self.badge, last_stack=2415, require_confirmation=True)
        self.assertIsNone(value)

    def test_legacy_fallback_remains_compatible(self):
        with patch.object(ocr_reader, "_resolve_image_input", return_value=None):
            self.assertEqual(ocr_reader.extract_hero_stack("absent.png", last_stack=2415), 2415)
            self.assertIsNone(ocr_reader.extract_hero_stack("absent.png", last_stack=2415, require_confirmation=True))


class TestLiveFinancialContext(unittest.TestCase):
    setUp = test_app_cache.TestPokerAnalyticsAppCache.setUp
    live = test_live_hands.TestLiveHands.live

    def test_live_oscillation_is_suspended_and_recovers(self):
        frame = test_live_hands.frame
        output = self.live([frame(stack=2415, pot=215), frame(stack=5, pot=295),
                            frame(stack=2375, pot=335)])
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual([c.kwargs["hero_stack"] for c in self.make_decision.call_args_list], [2415, 2375])
        self.assertIn("SEM RECOMENDAÇÃO ATUAL", output)
        self.assertFalse(self.analyses[1]["action_available"])

    def test_live_unknown_stack_is_not_replaced_by_previous_value(self):
        frame = test_live_hands.frame
        self.live([frame(stack=2415), frame(stack=None), frame(stack=2415)])
        self.calculate_equity.assert_called_once()
        self.assertIsNone(self.analyses[1]["financial_observation"]["hero_stack"])
        self.assertTrue(self.analyses[2]["action_available"])

    def test_live_unknown_pot_does_not_use_old_or_default_pot(self):
        frame = test_live_hands.frame
        self.live([frame(pot=215), frame(pot=None), frame(pot=215)])
        self.calculate_equity.assert_called_once()
        self.assertIsNone(self.analyses[1]["financial_observation"]["pot_size"])

    def test_live_requests_confirmed_stack_and_no_pot_fallback(self):
        with patch("main.PokerAnalyticsApp", return_value=self.app), \
                patch("main.grab_screen", side_effect=[object(), KeyboardInterrupt]), \
                patch("main.find_and_detect_hero_cards", return_value=(self.app.hero_cards, None)), \
                patch("main.detect_board", return_value=[]), \
                patch("main.extract_pot", return_value=215) as pot_ocr, \
                patch("main.detect_turn_and_bet_to_call", return_value=call_context(10)), \
                patch("main.extract_hero_stack", return_value=2415) as stack_ocr, \
                patch("main.extract_hero_position", return_value="BTN"), \
                patch("main.detect_active_opponents", return_value=2), \
                patch("main.time.sleep"), redirect_stdout(io.StringIO()):
            run_live()
        self.assertTrue(stack_ocr.call_args.kwargs["require_confirmation"])
        self.assertIsNone(pot_ocr.call_args.kwargs["default_value"])


if __name__ == "__main__":
    unittest.main()
