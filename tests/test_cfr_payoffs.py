"""4E-B2: utilidade financeira desde a raiz, sem bônus de showdown.

Os caminhos usam a contabilidade real da árvore 4E-B1. Respostas únicas em
alguns testes delimitam cenários explícitos, não uma nova política adversária.
"""

import io
import math
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.engine.cfr_solver import CFRSolver
from src.engine.evaluator import FULL_DECK, calculate_equity
from src.engine.game_tree import GameNode, GameTreeBuilder
from src.engine.range_model import OpponentModel
from src.engine.risk_engine import calculate_ev, make_decision
from tests import test_app_cache


HERO = ["Jh", "9d"]
BOARD = ["As", "Kd", "Qc", "7h", "2s"]


def root_node(**kwargs):
    values = dict(player=0, pot_size=100, hero_stack=500, villain_stack=480,
                  board_cards=BOARD, current_street="RIVER", bet_to_call=20)
    values.update(kwargs)
    return GameNode(**values)


def hero_utility(node, equity, initial_stack):
    """run_cfr retorna valor na perspectiva do jogador indicado no nó."""
    value = CFRSolver().run_cfr(node, HERO, equity, 1, 1, initial_stack)
    return value if node.player == 0 else -value


class TestFinancialPayoffs(unittest.TestCase):
    def setUp(self):
        self.builder = GameTreeBuilder(1)
        self.root = self.builder.build_tree(root_node())

    def test_fold_call_reproduction(self):
        self.assertEqual(hero_utility(self.root.children["FOLD"], 15, 500), 0)
        self.assertAlmostEqual(hero_utility(self.root.children["CALL"], 15, 500), -2)

    def test_showdown_win_loss_ties_and_fractional_equity(self):
        called = self.root.children["CALL"]  # P_final=120, investimento=20.
        for equity, expected in [(100, 100), (0, -20), (50, 40),
                                 (100/3, 20), (25, 10), (37.5, 25)]:
            with self.subTest(equity=equity):
                self.assertAlmostEqual(CFRSolver().compute_utility(called, equity, 500), expected)

    def test_call_positive_negative_and_break_even_match_shared_ev(self):
        for equity in (25, 15, 100/6):
            with self.subTest(equity=equity):
                expected = calculate_ev(pot_size=100, bet_to_call=20, equity=equity)
                self.assertAlmostEqual(hero_utility(self.root.children["CALL"], equity, 500), expected)

    def test_fold_before_investing_is_zero_independent_of_equity(self):
        for equity in (0, 15, 50, 100):
            self.assertEqual(hero_utility(self.root.children["FOLD"], equity, 500), 0)

    def test_fold_after_investing_loses_only_investment_since_root(self):
        free = root_node(bet_to_call=0)
        bet = self.builder._apply_action(free, "BET_25")
        raise_ = self.builder._apply_action(bet, "BET_50")
        folded = self.builder._apply_action(raise_, "FOLD")
        self.assertEqual(hero_utility(folded, 100, 500), -25)

    def test_previous_street_contribution_is_not_subtracted_twice(self):
        root = root_node(pot_size=200, bet_to_call=30, hero_contribution=20,
                         villain_contribution=50)
        called = self.builder.build_tree(root).children["CALL"]
        self.assertAlmostEqual(hero_utility(called, 50, 500), 85)

    def test_two_checks_have_no_cost_and_no_bonus(self):
        root = self.builder.build_tree(root_node(bet_to_call=0))
        showdown = root.children["CALL"].children["CALL"]
        self.assertTrue(showdown.is_terminal())
        self.assertEqual(showdown.hero_stack, 500)
        self.assertEqual(showdown.pot_size, 100)
        self.assertEqual(hero_utility(showdown, 25, 500), 25)
        self.assertEqual(hero_utility(showdown, 0, 500), 0)

    def test_bet_and_raise_called_payoffs_use_actual_cost(self):
        for call in (0, 20):
            with self.subTest(call=call):
                root = self.builder.build_tree(root_node(bet_to_call=call))
                raised = root.children["BET_25"]
                showdown = raised.children["CALL"]
                cost = raised.action_details["additional_amount"]
                self.assertEqual(cost, call + 25)
                self.assertAlmostEqual(hero_utility(showdown, 40, 500), .4 * showdown.pot_size - cost)

    def test_bet_and_raise_receive_fold_and_win_existing_pot(self):
        for call in (0, 20):
            with self.subTest(call=call):
                root = self.builder.build_tree(root_node(bet_to_call=call))
                folded = root.children["BET_25"].children["FOLD"]
                self.assertEqual(hero_utility(folded, 0, 500), 100)
                self.assertEqual(hero_utility(folded, 100, 500), 100)

    def test_all_in_showdown_closes_betting_and_conserves_chips(self):
        for hero_stack, villain_stack in ((100, 200), (200, 100), (100, 100)):
            with self.subTest(stacks=(hero_stack, villain_stack)):
                root = self.builder.build_tree(root_node(hero_stack=hero_stack,
                            villain_stack=villain_stack, bet_to_call=0, current_street="TURN"))
                showdown = root.children["ALL_IN"].children["CALL"]
                investment = min(hero_stack, villain_stack)
                self.assertTrue(showdown.is_terminal())
                self.assertFalse(showdown.children)
                self.assertEqual(showdown.pot_size, 100 + 2 * investment)
                self.assertEqual(showdown.hero_stack + showdown.villain_stack + showdown.pot_size,
                                 100 + hero_stack + villain_stack)
                self.assertAlmostEqual(hero_utility(showdown, 25, hero_stack),
                                       .25 * showdown.pot_size - investment)

    def test_short_call_uses_refunded_eligible_pot(self):
        root = self.builder.build_tree(root_node(hero_stack=5))
        showdown = root.children["CALL"]
        self.assertEqual(showdown.action_details["uncalled_refund"], 15)
        self.assertEqual(showdown.pot_size, 90)
        result = make_decision(40, 60, 0, 100, 20, 5, state="RIVER",
                               hero_cards=HERO, board_cards=BOARD, equity=40, num_opponents=1)
        self.assertAlmostEqual(hero_utility(showdown, 40, 5), .4 * 90 - 5)
        self.assertAlmostEqual(hero_utility(showdown, 40, 5), result["ev"])

    def test_zero_equity_small_all_in_cannot_gain_from_large_pot(self):
        root = self.builder.build_tree(root_node(pot_size=1000, bet_to_call=0,
                                                 hero_stack=10, villain_stack=100))
        called = root.children["ALL_IN"].children["CALL"]
        self.assertEqual(hero_utility(called, 0, 10), -10)

    def test_different_sizes_do_not_invert_financial_order(self):
        root = self.builder.build_tree(root_node(bet_to_call=0))
        smaller = hero_utility(root.children["BET_25"].children["CALL"], 46, 500)
        larger = hero_utility(root.children["BET_50"].children["CALL"], 46, 500)
        self.assertAlmostEqual(smaller, 44)
        self.assertAlmostEqual(larger, 42)
        self.assertGreater(smaller, larger)

    def test_no_bonus_at_multiple_pot_sizes(self):
        for pot in (0, 1, 100, 10000):
            with self.subTest(pot=pot):
                terminal = root_node(pot_size=pot, bet_to_call=0, current_street="SHOWDOWN")
                self.assertEqual(CFRSolver().compute_utility(terminal, 0, 500), 0)
                self.assertAlmostEqual(CFRSolver().compute_utility(terminal, 30, 500), .3 * pot)

    def test_terminal_perspective_sign_is_preserved(self):
        for player in (0, 1):
            terminal = root_node(hero_stack=480, villain_stack=480, pot_size=120,
                                 current_street="SHOWDOWN", bet_to_call=0, player=player)
            value = CFRSolver().run_cfr(terminal, HERO, 15, 1, 1, 500)
            self.assertAlmostEqual(value, -2 if player == 0 else 2)


class TestSolverFinancialOrdering(unittest.TestCase):
    def assert_valid_strategy(self, strategy, root):
        self.assertEqual(set(strategy), set(root.children))
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in strategy.values()))
        self.assertAlmostEqual(sum(strategy.values()), 1, delta=.0005 * len(strategy) + 1e-12)

    def test_fold_call_converges_to_financially_better_action(self):
        root = GameTreeBuilder(0).build_tree(root_node())
        self.assertEqual(set(root.children), {"FOLD", "CALL"})
        previous = 0
        for iterations in (10, 30, 100, 1000):
            with self.subTest(iterations=iterations):
                strategy = CFRSolver().solve(root, HERO, 15, iterations)
                self.assert_valid_strategy(strategy, root)
                self.assertGreater(strategy["FOLD"], strategy["CALL"])
                self.assertGreaterEqual(strategy["FOLD"], previous)
                previous = strategy["FOLD"]
        self.assertGreater(previous, .99)

    def test_solver_is_deterministic(self):
        root = GameTreeBuilder(0).build_tree(root_node())
        self.assertEqual(CFRSolver().solve(root, HERO, 15, 30),
                         CFRSolver().solve(root, HERO, 15, 30))

    def test_bluff_with_no_fold_equity_loses_and_with_sufficient_folds_profits(self):
        root = GameTreeBuilder(1).build_tree(root_node(bet_to_call=0))
        bet = root.children["BET_25"]
        win = hero_utility(bet.children["FOLD"], 0, 500)
        lose = hero_utility(bet.children["CALL"], 0, 500)
        self.assertEqual((win, lose), (100, -25))
        self.assertLess(lose, 0)  # Frequência de fold zero.
        self.assertEqual(.2 * win + .8 * lose, 0)
        self.assertEqual(.4 * win + .6 * lose, 25)

    def test_solver_prefers_bluff_only_when_specified_response_makes_it_profitable(self):
        for response, expected_best in (("CALL", "CALL"), ("FOLD", "BET_25")):
            with self.subTest(opponent_response=response):
                root = GameTreeBuilder(1).build_tree(root_node(bet_to_call=0))
                check = root.children["CALL"]
                check.children = {"CALL": check.children["CALL"]}
                bet = root.children["BET_25"]
                bet.children = {response: bet.children[response]}
                root.children = {"CALL": check, "BET_25": bet}
                strategy = CFRSolver().solve(root, HERO, 0, 100)
                self.assert_valid_strategy(strategy, root)
                other = "BET_25" if expected_best == "CALL" else "CALL"
                self.assertGreater(strategy[expected_best], strategy[other])


class TestPayoffIntegration(unittest.TestCase):
    def test_royal_equity_reaches_real_solver_cache_and_panel_without_half_tie_conversion(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = ["2h", "3h"]
        self.app.num_opponents = 3
        self.app.opponent_model = OpponentModel(FULL_DECK)
        board = ["10s", "Js", "Qs", "Ks", "As"]
        previous_random = random.getstate()
        self.addCleanup(random.setstate, previous_random)
        random.seed(20261011)
        original = CFRSolver.compute_utility
        evaluations = []

        def traced(solver, terminal, equity, initial):
            actual = original(solver, terminal, equity, initial)
            evaluations.append((equity, actual,
                                terminal.hero_stack + .25 * terminal.pot_size - initial))
            return actual

        context = dict(is_hero_turn=True, action_type="CALL", bet_to_call=20,
                       available_actions=["FOLD", "CALL", "RAISE"])
        output = io.StringIO()
        with patch("main.calculate_equity", wraps=calculate_equity) as simulation, \
             patch("main.make_decision", wraps=make_decision) as decision, \
             patch.object(CFRSolver, "compute_utility", traced), redirect_stdout(output):
            first = self.app.process_cards(board, 120, 20, True, action_context=context)
            cached = self.app.process_cards(board, 120, 20, True, action_context=context)
        self.assertTrue(evaluations)
        for equity, actual, expected in evaluations:
            self.assertEqual(equity, 25)
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(first["p_tie"], 100)
        self.assertEqual(first["equity"], 25)
        self.assertEqual(first["ev"], 15)
        self.assertIs(first, cached)
        simulation.assert_called_once()
        decision.assert_called_once()
        self.assertRegex(output.getvalue(), r"Equity Consolidada:\s+25\.00%")
        self.assertIn("Ramo avaliado", output.getvalue())
        if first["action"] in ("BET", "RAISE", "ALL-IN"):
            self.assertFalse(first["action_available"])
            self.assertIn("Investimento adicional", output.getvalue())


if __name__ == "__main__":
    unittest.main()
