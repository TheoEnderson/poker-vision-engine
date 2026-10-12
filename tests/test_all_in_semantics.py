"""Valor impresso em ALL-IN não confirma custo incremental ou pote elegível."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import main
from src.ocr.ocr_reader import parse_action_button_text
from tests import test_financial_context as finance
from tests.test_hero_allin_session import hero_frame


class TestAllInSemantics(unittest.TestCase):
    setUp = finance.TestFinancialContext.setUp
    process = finance.TestFinancialContext.process
    assert_suspended = finance.TestFinancialContext.assert_suspended

    def test_1250_total_is_reported_as_unsupported_not_visual_failure(self):
        context = parse_action_button_text("DESISTO ALL-IN 1.250")
        result = self.process(1240, pot=2045, call=context["bet_to_call"], context=context)
        self.assert_suspended(result)
        self.assertIsNone(result["bet_to_call"])
        self.assertEqual(result.get("financial_status"), "UNSUPPORTED_ALL_IN")
        output = io.StringIO()
        with redirect_stdout(output):
            self.app.render_dashboard(result)
        self.assertIn("CENÁRIO FINANCEIRO NÃO SUPORTADO", output.getvalue())
        self.assertIn("1250", output.getvalue())
        self.assertNotIn("CAPTURA INCERTA", output.getvalue())
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()
        self.assertIsNone(self.app.last_analysis)

    def test_live_real_detector_recovers_borders_but_all_in_remains_suspended(self):
        self.app.hero_cards = ['5h', '7c']
        context = parse_action_button_text("DESISTO ALL-IN 1.250")
        unreadable = hero_frame(116)
        unreadable[340:364, 1556:1635] = 0
        frames = [hero_frame(), hero_frame(106), hero_frame(116), unreadable, hero_frame()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.jsonl'
            with patch('main.PokerAnalyticsApp', return_value=self.app), \
                 patch('main.grab_screen', side_effect=frames + [KeyboardInterrupt]), \
                 patch('main.detect_board', return_value=['Vazio'] * 5), \
                 patch('main.extract_pot', return_value=2045), \
                 patch('main.extract_hand_identity', return_value=None), \
                 patch('main.extract_hero_position', return_value='BTN'), \
                 patch('main.detect_turn_and_bet_to_call', return_value=context), \
                 patch('main.detect_active_opponents', return_value=3), \
                 patch('main.extract_hero_stack', return_value=1240), \
                 patch('main.time.sleep'), redirect_stdout(io.StringIO()):
                main.run_live(profile_jsonl=path)
            records = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in records if r['type'] == 'cycle']
        self.assertEqual([r['outcome'] for r in rows],
                         ['financial_suspension'] * 3 + ['recognition_unconfirmed', 'financial_suspension'])
        self.assertTrue(rows[1]['hero_detection']['separated_cards'])
        self.assertTrue(rows[2]['hero_detection']['separated_cards'])
        self.assertEqual(rows[-1]['hero_observed'], ['5h', '7c'])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertIsNone(self.app.last_analysis)
        self.assertEqual(self.app.opponent_model.mock_calls, [])
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()

    def test_displayed_total_with_previous_contribution_is_not_call_cost(self):
        context = parse_action_button_text("DESISTO ALL-IN 1.195")
        self.assertIsNone(context["bet_to_call"])
        self.assertEqual(context["all_in_displayed_amount"], 1195)
        self.assertEqual(context["action_type"], "ALL-IN")
        result = self.process(1190, pot=2039, call=None, context=context)
        self.assert_suspended(result)
        self.assertIn("pote elegível", result["reason"])
        self.assertNotIn("CALL de 1195", result["reason"])
        self.calculate_equity.assert_not_called()
        self.make_decision.assert_not_called()

    def test_even_matching_stack_does_not_establish_eligible_pot(self):
        context = parse_action_button_text("FOLD ALL-IN 1190")
        result = self.process(1190, pot=2039, call=context["bet_to_call"], context=context)
        self.assert_suspended(result)
        self.calculate_equity.assert_not_called()

    def test_unreadable_all_in_amount_remains_unknown(self):
        context = parse_action_button_text("FOLD ALL-IN ???")
        self.assertIsNone(context["bet_to_call"])
        self.assertIsNone(context["all_in_displayed_amount"])
        result = self.process(1190, pot=2039, call=None, context=context)
        self.assert_suspended(result)

    def test_normal_call_next_to_all_in_keeps_its_own_incremental_amount(self):
        context = parse_action_button_text("FOLD CALL 15 ALL-IN 1195")
        self.assertEqual(context["bet_to_call"], 15)
        result = self.process(1190, pot=55, call=15, context=context)
        self.assertTrue(result["action_available"])


if __name__ == "__main__":
    unittest.main()
