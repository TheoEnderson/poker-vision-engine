"""Valor impresso em ALL-IN não confirma custo incremental ou pote elegível."""
import unittest
from src.ocr.ocr_reader import parse_action_button_text
from tests import test_financial_context as finance


class TestAllInSemantics(unittest.TestCase):
    setUp = finance.TestFinancialContext.setUp
    process = finance.TestFinancialContext.process
    assert_suspended = finance.TestFinancialContext.assert_suspended

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
