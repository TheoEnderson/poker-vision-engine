"""Regressões 4E-B1: ledger da street, all-in e rastreabilidade de tamanhos."""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import PokerAnalyticsApp
from src.engine.game_tree import GameNode, GameTreeBuilder
from src.engine.cfr_solver import CFRSolver
from src.engine.risk_engine import make_decision
from tests import test_app_cache


BOARD = ["Qs", "7s", "6s", "Jd", "6d"]
HERO = ["10s", "Jh"]


def node(**kwargs):
    values = dict(player=0, pot_size=100, hero_stack=500, villain_stack=450,
                  board_cards=BOARD, current_street="RIVER", bet_to_call=50)
    values.update(kwargs)
    return GameNode(**values)


class TestBettingLedger(unittest.TestCase):
    def setUp(self):
        self.builder = GameTreeBuilder(1)

    def test_percent_raise_adds_call_instead_of_underpaying_it(self):
        root = node()
        for action, expected in [("BET_25", 75), ("BET_50", 100), ("BET_100", 150)]:
            with self.subTest(action=action):
                child = self.builder._apply_action(root, action)
                self.assertEqual(root.hero_stack - child.hero_stack, expected)
                self.assertGreater(expected, root.bet_to_call)

    def test_opponent_pays_only_difference_to_raise_total(self):
        raised = self.builder._apply_action(node(), "BET_50")
        self.assertEqual(raised.bet_to_call, 50)
        called = self.builder._apply_action(raised, "CALL")
        self.assertEqual((called.pot_size, called.hero_stack, called.villain_stack), (250, 400, 400))

    def test_reraise_preserves_previous_contributions(self):
        root = node(hero_contribution=0, villain_contribution=50, min_raise_increment=50)
        raised = self.builder._apply_action(root, "BET_50")
        reraised = self.builder._apply_action(raised, "BET_25")
        self.assertEqual(reraised.action_details["additional_amount"], 100)
        self.assertEqual(reraised.action_details["raise_to"], 150)
        self.assertEqual((reraised.bet_to_call, reraised.min_raise_increment), (50, 50))
        called = self.builder._apply_action(reraised, "CALL")
        self.assertEqual((called.hero_contribution, called.villain_contribution), (150, 150))
        self.assertEqual((called.pot_size, called.hero_stack, called.villain_stack), (350, 350, 350))

    def test_explicit_contributions_distinguish_total_and_additional(self):
        root = node(pot_size=200, bet_to_call=30, hero_contribution=20,
                    villain_contribution=50, min_raise_increment=50)
        raised = self.builder._apply_action(root, "BET_25")
        self.assertEqual(raised.hero_contribution, 100)
        self.assertEqual(raised.villain_contribution, 50)
        self.assertEqual(raised.bet_to_call, 50)
        self.assertEqual(raised.action_details["additional_amount"], 80)
        self.assertEqual(raised.action_details["raise_to"], 100)
        self.assertEqual(raised.action_details["raise_by"], 50)
        self.assertTrue(raised.action_details["size_verified"])

    def test_minimum_raise_excludes_under_minimum_candidates(self):
        root = node(hero_contribution=0, villain_contribution=50, min_raise_increment=50)
        actions = self.builder._get_valid_actions(root, 0)
        self.assertNotIn("BET_25", actions)
        self.assertIn("BET_50", actions)
        with self.assertRaises(ValueError):
            self.builder._apply_action(root, "BET_25")

    def test_unknown_minimum_and_contributions_are_not_certified(self):
        child = self.builder._apply_action(node(), "BET_25")
        self.assertFalse(child.action_details["size_verified"])
        self.assertIsNone(child.action_details["raise_to"])
        self.assertIn("mínimo", child.action_details["validation_reason"])
        self.assertIsNone(child.action_details["short_raise"])

    def test_check_and_fold_legality(self):
        free = node(bet_to_call=0)
        self.assertNotIn("FOLD", self.builder._get_valid_actions(free, 0))
        with self.assertRaises(ValueError):
            self.builder._apply_action(free, "FOLD")
        checked = self.builder._apply_action(free, "CALL")
        self.assertEqual(checked.action_details["action_type"], "CHECK")
        self.assertEqual(checked.action_details["additional_amount"], 0)
        self.assertEqual((checked.pot_size, checked.hero_stack), (100, 500))
        facing = node()
        self.assertIn("FOLD", self.builder._get_valid_actions(facing, 0))
        folded = self.builder._apply_action(facing, "FOLD")
        self.assertTrue(folded.is_terminal())
        self.assertEqual((folded.pot_size, folded.hero_stack), (100, 500))

    def test_call_uses_only_additional_cost(self):
        called = self.builder._apply_action(node(), "CALL")
        self.assertEqual((called.pot_size, called.hero_stack), (150, 450))
        self.assertEqual(called.hero_contribution, called.villain_contribution)
        self.assertEqual(called.action_details["additional_amount"], 50)

    def test_initial_bet_and_known_minimum(self):
        root = node(bet_to_call=0, hero_contribution=0, villain_contribution=0,
                    min_raise_increment=30)
        self.assertNotIn("BET_25", self.builder._get_valid_actions(root, 0))
        bet = self.builder._apply_action(root, "BET_50")
        self.assertEqual(bet.action_details["action_type"], "BET")
        self.assertEqual((bet.hero_stack, bet.pot_size, bet.bet_to_call), (450, 150, 50))

    def test_raise_that_does_not_fit_stack_is_not_silently_clamped(self):
        root = node(hero_stack=60)
        self.assertNotIn("BET_25", self.builder._get_valid_actions(root, 0))
        with self.assertRaises(ValueError):
            self.builder._apply_action(root, "BET_25")

    def test_zero_pot_cannot_generate_zero_sized_bets(self):
        root = node(pot_size=0, bet_to_call=0)
        self.assertFalse(any(a.startswith("BET_") for a in self.builder._get_valid_actions(root, 0)))

    def test_raise_equivalent_to_call_is_rejected(self):
        # Na abstração proporcional, P=0 produziria aumento zero além do call.
        root = node(pot_size=0, bet_to_call=50)
        self.assertNotIn("BET_25", self.builder._get_valid_actions(root, 0))
        with self.assertRaises(ValueError):
            self.builder._apply_action(root, "BET_25")

    def test_contradictory_contributions_are_rejected(self):
        with self.assertRaises(ValueError):
            node(hero_contribution=20, villain_contribution=50, bet_to_call=50)

    def test_street_advancement_resets_contributions(self):
        root = self.builder.build_tree(node(current_street="FLOP"))
        called = root.children["CALL"]
        self.assertEqual(called.current_street, "TURN")
        self.assertEqual((called.hero_contribution, called.villain_contribution), (0, 0))
        self.assertEqual(called.bet_to_call, 0)


class TestAllInFinances(unittest.TestCase):
    def setUp(self):
        self.builder = GameTreeBuilder(1)

    def test_all_in_call_closes_betting_before_future_streets(self):
        root = self.builder.build_tree(node(hero_stack=100, villain_stack=200,
                                            bet_to_call=0, current_street="TURN", board_cards=BOARD[:4]))
        resolved = root.children["ALL_IN"].children["CALL"]
        self.assertEqual((resolved.hero_stack, resolved.villain_stack, resolved.pot_size), (0, 100, 300))
        self.assertTrue(resolved.is_terminal())
        self.assertEqual(resolved.current_street, "SHOWDOWN")
        self.assertFalse(resolved.children)
        self.assertEqual(resolved.board_cards, BOARD[:4])

    def test_no_raise_against_opponent_already_all_in(self):
        facing = node(hero_stack=200, villain_stack=0, bet_to_call=100)
        self.assertEqual(self.builder._get_valid_actions(facing, 0), ["FOLD", "CALL"])
        with self.assertRaises(ValueError):
            self.builder._apply_action(facing, "ALL_IN")

    def test_short_all_in_raise_is_distinct_from_full_raise(self):
        root = node(hero_stack=70, hero_contribution=0, villain_contribution=50,
                    min_raise_increment=50)
        raised = self.builder._apply_action(root, "ALL_IN")
        self.assertTrue(raised.action_details["is_all_in"])
        self.assertTrue(raised.action_details["short_raise"])
        self.assertEqual(raised.action_details["raise_by"], 20)
        self.assertEqual(raised.bet_to_call, 20)
        self.assertEqual(raised.min_raise_increment, 50)

    def test_short_call_refunds_unmatched_chips_in_heads_up(self):
        root = self.builder.build_tree(node(hero_stack=20, villain_stack=450,
                                            hero_contribution=0, villain_contribution=50))
        called = root.children["CALL"]
        self.assertEqual(called.action_details["additional_amount"], 20)
        self.assertEqual(called.action_details["uncalled_refund"], 30)
        self.assertEqual((called.pot_size, called.hero_stack, called.villain_stack), (90, 0, 480))
        self.assertTrue(called.is_terminal())

    def test_overbet_all_in_refunds_excess_instead_of_contesting_it(self):
        root = self.builder.build_tree(node(hero_stack=200, villain_stack=100, bet_to_call=0))
        called = root.children["ALL_IN"].children["CALL"]
        self.assertEqual((called.pot_size, called.hero_stack, called.villain_stack), (300, 100, 0))
        self.assertEqual(called.action_details["uncalled_refund"], 100)
        self.assertTrue(called.is_terminal())

    def test_equal_stacks_all_in_and_call(self):
        root = self.builder.build_tree(node(hero_stack=100, villain_stack=100, bet_to_call=0))
        called = root.children["ALL_IN"].children["CALL"]
        self.assertEqual((called.hero_stack, called.villain_stack, called.pot_size), (0, 0, 300))
        self.assertTrue(called.is_terminal())

    def test_every_branch_conserves_chips_and_nonnegative_stacks(self):
        for h, v, call in [(500, 450, 50), (100, 200, 0), (200, 100, 0), (20, 450, 50)]:
            with self.subTest(stacks=(h, v), call=call):
                root = self.builder.build_tree(node(hero_stack=h, villain_stack=v, bet_to_call=call))
                total = 100 + h + v
                def check(n):
                    self.assertAlmostEqual(n.pot_size + n.hero_stack + n.villain_stack, total)
                    self.assertGreaterEqual(min(n.pot_size, n.hero_stack, n.villain_stack), 0)
                    for child in n.children.values():
                        check(child)
                check(root)

    def test_financial_payoff_and_cfr_update_remain_compatible(self):
        root = self.builder.build_tree(node())
        strategy = CFRSolver().solve(root, HERO, 12.99, 30)
        self.assertEqual(set(strategy), set(root.children))
        self.assertAlmostEqual(sum(strategy.values()), 1, delta=.003)
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in strategy.values()))
        called = root.children["CALL"]
        actual = CFRSolver().compute_utility(called, 50, 500)
        self.assertAlmostEqual(actual, called.hero_stack + .5 * called.pot_size - 500)


class TestActionPresentation(unittest.TestCase):
    def test_partial_multiway_call_is_explicitly_unsupported(self):
        with patch("src.engine.cfr_solver.CFRSolver.solve") as solve:
            result = make_decision(40, 60, 0, 100, 50, 20, state="RIVER",
                                   hero_cards=HERO, board_cards=BOARD, equity=40, num_opponents=2)
        self.assertFalse(result["action_details"]["supported"])
        self.assertIn("multiway", result["reason"])
        self.assertEqual(result["recommended_amount"], 0)
        solve.assert_not_called()
        result.update(hero_stack=20, recalculated=True)
        PokerAnalyticsApp._with_action_context(result, dict(is_hero_turn=True,
                         action_type="CALL", bet_to_call=50, available_actions=["CALL"]))
        self.assertFalse(result["action_available"])

    def test_unrepresentable_refund_cannot_fall_back_to_validated_action(self):
        result = make_decision(40, 60, 0, 5, 50, 20, state="RIVER", hero_cards=HERO,
                               board_cards=BOARD, equity=40)
        self.assertFalse(result["action_details"]["supported"])
        self.assertIn("devolução", result["reason"])

    def test_solver_cannot_present_an_action_without_a_matching_branch(self):
        with patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"BET_25": 1}):
            result = make_decision(40, 60, 0, 100, 50, 500, state="RIVER", hero_cards=HERO,
                                   board_cards=BOARD, equity=40, min_raise_increment=50)
        self.assertFalse(result["action_details"]["supported"])

    def test_historical_size_is_exactly_the_branch_investment(self):
        with patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"BET_25": .881, "BET_50": .119}) as solve:
            result = make_decision(12.99, 87.01, 0, 1315, 1000, 2375, state="RIVER",
                                   hero_cards=HERO, board_cards=BOARD, equity=12.99)
        root = solve.call_args.args[0]
        branch = root.children["BET_25"]
        self.assertEqual(result["recommended_amount"], root.hero_stack - branch.hero_stack)
        self.assertEqual(result["recommended_amount"], 1328.75)
        self.assertEqual(result["selected_action"], "BET_25")
        self.assertEqual(result["selected_action_probability"], .881)
        self.assertEqual(result["action_details"], branch.action_details)
        self.assertIn("BET_25", result["reason"])
        self.assertIn("BET_50", result["reason"])
        self.assertNotIn("RAISE:1.00", result["reason"])
        self.assertEqual(solve.call_args.kwargs["iterations"], 30)

    def test_known_total_and_additional_are_returned_separately(self):
        with patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"BET_25": 1}):
            result = make_decision(50, 50, 0, 200, 30, 500, state="RIVER", hero_cards=HERO,
                                   board_cards=BOARD, equity=50, hero_contribution=20,
                                   villain_contribution=50, min_raise_increment=50, villain_stack=450)
        self.assertEqual(result["recommended_amount"], 80)
        self.assertEqual(result["action_details"]["raise_to"], 100)
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = HERO
        self.app.hero_stack = 500
        output = io.StringIO()
        with patch("main.make_decision", return_value=result), redirect_stdout(output):
            self.app.process_cards(BOARD, 200, 30, True)
        self.assertIn("Investimento adicional (modelo): 80", output.getvalue())
        self.assertIn("Total da aposta na street: 100", output.getvalue())

    def test_uncertain_minimum_stays_informative_with_raise_button(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = HERO
        self.app.hero_stack = 2375
        context = dict(is_hero_turn=True, bet_to_call=1000, action_type="CALL",
                       available_actions=["FOLD", "CALL", "RAISE"])
        output = io.StringIO()
        with patch("main.make_decision", wraps=make_decision), \
             patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"BET_25": 1}), \
             redirect_stdout(output):
            first = self.app.process_cards(BOARD, 1315, 1000, True, action_context=context)
            cached = self.app.process_cards(BOARD, 1315, 1000, True, action_context=context)
        self.assertIs(first, cached)
        self.assertFalse(first["action_available"])
        self.assertEqual(first["action_details"]["additional_amount"], 1328.75)
        self.assertIn("Investimento adicional", output.getvalue())
        self.assertIn("1328.75", output.getvalue())
        self.assertIn("Total da street não confirmado", output.getvalue())
        self.assertNotIn("RECOMENDAÇÃO PARA A CAPTURA:", output.getvalue())
        self.calculate_equity.assert_called_once()


if __name__ == "__main__":
    unittest.main()
