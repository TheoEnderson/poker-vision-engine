"""EV líquido de um único pote, sem rake, apostas futuras ou side pots."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.engine.risk_engine import calculate_ev, calculate_pot_odds, make_decision
from tests import test_app_cache


class TestCallEV(unittest.TestCase):
    def test_exact_equities_and_signs(self):
        for equity, expected in [(100, 120), (0, -20), (50, 50),
                                 (100 / 3, 80 / 3), (25, 15), (100 / 9, -40 / 9), (14.94, .916)]:
            with self.subTest(equity=equity):
                self.assertAlmostEqual(calculate_ev(pot_size=120, bet_to_call=20, equity=equity), expected)

    def test_break_even_matches_pot_odds(self):
        self.assertEqual(calculate_ev(pot_size=60, bet_to_call=20, equity=25), 0)
        self.assertAlmostEqual(calculate_ev(pot_size=120, bet_to_call=20,
                                           equity=calculate_pot_odds(20, 120)), 0)

    def test_zero_call_is_only_expected_pot_share(self):
        self.assertEqual(calculate_ev(pot_size=120, bet_to_call=0, equity=25), 30)
        self.assertEqual(calculate_ev(pot_size=0, bet_to_call=0, equity=100), 0)

    def test_percentages_are_not_fractions(self):
        self.assertEqual(calculate_ev(pot_size=100, bet_to_call=0, equity=1), 1)
        self.assertEqual(calculate_ev(pot_size=100, bet_to_call=0, equity=100), 100)

    def test_required_examples(self):
        self.assertAlmostEqual(calculate_ev(14.09, 84.21, 1.70, 120, 20, num_opponents=1), .916)
        self.assertAlmostEqual(calculate_ev(pot_size=1315, bet_to_call=1000, equity=12.99), -699.2815)
        self.assertEqual(calculate_ev(0, 0, 100, 120, 20, equity=25), 15)

    def test_mixed_outcomes_use_authoritative_equity_not_half_ties(self):
        # Uma vitória, uma derrota, três empates entre 2, 3 e 4 vencedores.
        equity = 100 * (1 + 1/2 + 1/3 + 1/4) / 5
        self.assertAlmostEqual(calculate_ev(20, 20, 60, 120, 20, equity=equity),
                               (120 - 20 + 50 + 80/3 + 15) / 5)

    def test_legacy_without_ties_is_exact(self):
        self.assertEqual(calculate_ev(40, 60, 0, 120, 20), 36)

    def test_legacy_ties_require_equity_or_explicit_heads_up(self):
        for opponents in (None, 2, 3, 8):
            with self.subTest(opponents=opponents), self.assertRaisesRegex(ValueError, "equity"):
                calculate_ev(0, 0, 100, 120, 20, num_opponents=opponents)
        self.assertEqual(calculate_ev(0, 0, 100, 120, 20, num_opponents=1), 50)

    def test_invalid_or_missing_values_are_rejected(self):
        base = dict(pot_size=120, bet_to_call=20, equity=25)
        for key, invalid_values in [("equity", [None, -1, 101, float("nan"), float("inf"), True, "25"]),
                                    ("pot_size", [None, -1, float("nan"), float("inf")]),
                                    ("bet_to_call", [None, -1, float("nan"), float("inf")])]:
            for value in invalid_values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    calculate_ev(**{**base, key: value})
        for probabilities in [(40, 50, 0), (None, 60, 0), (-1, 101, 0), (float("nan"), 60, 0)]:
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                calculate_ev(*probabilities, 120, 20)

    def test_risk_engine_uses_shared_formula_without_changing_selected_action(self):
        with patch("src.engine.blueprint.PreflopBlueprint.sample_action", return_value="ALL_IN"):
            for equity in (0, 25, 100):
                result = make_decision(0, 0, 100, 120, 20, 1000, equity=equity)
                self.assertEqual(result["ev"], calculate_ev(pot_size=120, bet_to_call=20, equity=equity))
                self.assertEqual(result["action"], "ALL-IN")
                self.assertEqual(result["recommended_amount"], 1000)

    def test_risk_engine_rejects_ambiguous_legacy_tie_before_strategy(self):
        with patch("src.engine.blueprint.PreflopBlueprint.sample_action") as strategy:
            with self.assertRaisesRegex(ValueError, "equity"):
                make_decision(40, 58, 2, 120, 20, 1000)
            strategy.assert_not_called()

    def test_risk_engine_legacy_no_tie_and_explicit_heads_up(self):
        with patch("src.engine.blueprint.PreflopBlueprint.sample_action", return_value="CALL"):
            self.assertEqual(make_decision(40, 60, 0, 120, 20, 1000)["ev"], 36)
            self.assertEqual(make_decision(0, 0, 100, 120, 20, 1000, num_opponents=1)["ev"], 50)


class TestEVIntegration(unittest.TestCase):
    def setUp(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.calculate_equity.return_value = (14.09, 1.70, 84.21, 14.94, .001)

    def test_call_ev_reaches_cache_dashboard_and_dataset(self):
        output = io.StringIO()
        with patch("main.make_decision", wraps=make_decision), \
             patch("src.engine.blueprint.PreflopBlueprint.sample_action", return_value="CALL"), \
             redirect_stdout(output):
            first = self.app.process_cards([], 120, 20, True)
            second = self.app.process_cards([], 120, 20, True)
        self.assertAlmostEqual(first["ev"], .916)
        self.assertIs(first, second)
        self.calculate_equity.assert_called_once()
        self.assertIn("EV do CALL", output.getvalue())
        self.assertRegex(output.getvalue(), r"\+0\.92 fichas")
        self.assertEqual(self.app.dataset_logger.log_state.call_args.args[0]["ev"], first["ev"])

    def test_zero_call_label_does_not_claim_full_check_ev(self):
        output = io.StringIO()
        with patch("main.make_decision", wraps=make_decision), \
             patch("src.engine.blueprint.PreflopBlueprint.sample_action", return_value="CALL"), \
             redirect_stdout(output):
            result = self.app.process_cards([], 120, 0, True)
        self.assertAlmostEqual(result["ev"], .1494 * 120)
        self.assertIn("Parcela esperada do pote", output.getvalue())
        self.assertIn("não é EV estratégico de CHECK", output.getvalue())
        self.assertNotIn("EV de Call/Check", output.getvalue())

    def test_raise_keeps_call_reference_clearly_labeled(self):
        output = io.StringIO()
        with patch("main.make_decision", wraps=make_decision), \
             patch("src.engine.blueprint.PreflopBlueprint.sample_action", return_value="BET_50"), \
             redirect_stdout(output):
            result = self.app.process_cards([], 120, 20, True)
        self.assertEqual(result["action"], "RAISE")
        self.assertAlmostEqual(result["ev"], .916)
        self.assertIn("não é EV de RAISE/ALL-IN", output.getvalue())


if __name__ == "__main__":
    unittest.main()
