"""4E-B3: precisão, solução conhecida e limites do domínio abstrato."""

import io
import math
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.engine.cfr_solver import CFRSolver
from src.engine.risk_engine import make_decision
from tests import test_app_cache
from tests.cfr_scenarios import (HERO, ITERATIONS, MATRIX, average_policy, count_nodes,
                                 full_information_gap, matrix_metrics, matrix_root,
                                 MatrixSolver, poker_root, scenarios)


class TestCFRPrecision(unittest.TestCase):
    def assert_policy(self, strategy, root):
        self.assertEqual(set(strategy), set(root.children))
        self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in strategy.values()))
        self.assertAlmostEqual(math.fsum(strategy.values()), 1, delta=1e-12)

    def test_six_uniform_actions_are_not_rounded_inside_solver(self):
        root = poker_root()
        self.assertEqual(len(root.children), 6)
        result = CFRSolver().solve(root, HERO, 40, 1)
        self.assert_policy(result, root)
        self.assertTrue(all(abs(p - 1/6) < 1e-15 for p in result.values()))

    def test_strategy_matches_unrounded_accumulated_average(self):
        root = poker_root()
        solver = CFRSolver()
        result = solver.solve(root, HERO, 40, 30)
        self.assert_policy(result, root)
        expected = average_policy(solver, root)
        for action in result:
            self.assertAlmostEqual(result[action], expected[action], delta=1e-14)

    def test_zero_iterations_returns_normalized_uniform_policy(self):
        root = poker_root()
        self.assert_policy(CFRSolver().solve(root, HERO, 40, 0), root)
        solver = CFRSolver()
        solver.cumulative_strategy[root.get_info_set(HERO)] = dict.fromkeys(root.children, 0.)
        self.assert_policy(solver.solve(root, HERO, 40, 0), root)

    def test_terminal_has_no_actions(self):
        root = poker_root(depth=0).children["CALL"]
        self.assertEqual(CFRSolver().solve(root, HERO, 15, 0), {})

    def test_regret_matching_clamps_negative_regrets_and_normalizes(self):
        solver = CFRSolver()
        solver.cumulative_regrets["I"] = {"A": -2, "B": 1, "C": 3}
        self.assertEqual(solver.get_strategy("I", ["A", "B", "C"]), {"A": 0, "B": .25, "C": .75})
        solver.cumulative_regrets["I"] = {"A": -2, "B": -1}
        self.assertEqual(solver.get_strategy("I", ["A", "B"]), {"A": .5, "B": .5})

    def test_reach_weights_and_nonnegative_regret_update_are_preserved(self):
        root = poker_root(depth=0)
        solver = CFRSolver()
        value = solver.run_cfr(root, HERO, 15, .2, .7, 500)
        info = root.get_info_set(HERO)
        self.assertAlmostEqual(value, -1)
        self.assertEqual(solver.cumulative_strategy[info], {"FOLD": .1, "CALL": .1})
        self.assertEqual(solver.cumulative_regrets[info], {"FOLD": .7, "CALL": 0})

    def test_large_finite_weights_do_not_overflow_normalization(self):
        root = poker_root(depth=0)
        solver = CFRSolver()
        solver.cumulative_strategy[root.get_info_set(HERO)] = {"FOLD": 1e308, "CALL": 1e308}
        self.assertEqual(solver.solve(root, HERO, 15, 0), {"FOLD": .5, "CALL": .5})
        solver.cumulative_regrets["I"] = {"A": 1e308, "B": 1e308}
        self.assertEqual(solver.get_strategy("I", ["A", "B"]), {"A": .5, "B": .5})

    def test_invalid_average_weights_are_rejected_not_returned_as_probabilities(self):
        root = poker_root(depth=0)
        for invalid in (float("nan"), float("inf"), -1):
            solver = CFRSolver()
            solver.cumulative_strategy[root.get_info_set(HERO)] = {"FOLD": invalid, "CALL": 1}
            with self.subTest(weight=invalid), self.assertRaises(ValueError):
                solver.solve(root, HERO, 15, 0)


class TestKnownConvergence(unittest.TestCase):
    def test_matrix_reference_solution_and_gap_measure_are_exact(self):
        p, q, value = .6, .4, -.2
        for row in MATRIX:
            self.assertAlmostEqual(q * row[0] + (1-q) * row[1], value)
        for j in range(2):
            self.assertAlmostEqual(p * MATRIX[0][j] + (1-p) * MATRIX[1][j], value)
        root, solver = matrix_root(), MatrixSolver()
        solver.cumulative_strategy = {"ROW": {"R0": p, "R1": 1-p},
                                      "COLUMN": {"C0": q, "C1": 1-q}}
        self.assertAlmostEqual(matrix_metrics(solver, root)["nash_conv"], 0, delta=1e-12)
        solver.cumulative_strategy = {"ROW": {"R0": 1., "R1": 0.},
                                      "COLUMN": {"C0": 1., "C1": 0.}}
        self.assertEqual(matrix_metrics(solver, root)["nash_conv"], 2)

    def test_known_pure_actions_and_response_scenarios(self):
        cases = scenarios()
        for name, best, gap_bound in (("call_negative", "FOLD", 1.),
                                      ("call_positive", "CALL", 5.),
                                      ("bet_always_called", "CALL", 12.5),
                                      ("bet_always_folded", "BET_25", 50.)):
            root, factory, equity = cases[name]
            for iterations in ITERATIONS:
                with self.subTest(scenario=name, iterations=iterations):
                    solver = factory()
                    strategy = solver.solve(root, HERO, equity, iterations)
                    self.assertEqual(max(strategy, key=strategy.get), best)
                    # Duas ações e um único agente com escolha: massa exploratória
                    # inicial 1/(2N), logo perda exata |U1-U2|/(2N).
                    self.assertAlmostEqual(full_information_gap(root, solver, equity),
                                           gap_bound / iterations, delta=1e-10)

    def test_mixed_game_matches_known_solution_and_reduces_nash_conv(self):
        metrics = []
        for iterations in ITERATIONS:
            root = matrix_root()
            solver = MatrixSolver()
            solver.solve(root, HERO, 0, iterations)
            result = matrix_metrics(solver, root)
            metrics.append(result)
            self.assertGreaterEqual(result["nash_conv"], -1e-12)
        # Solução exata: p(linha 0)=3/5, q(coluna 0)=2/5, valor=-1/5.
        self.assertLess(metrics[-1]["row_l1"], .05)
        self.assertLess(metrics[-1]["column_l1"], .05)
        self.assertLess(metrics[-1]["nash_conv"], .05)
        self.assertLess(metrics[-1]["nash_conv"], metrics[0]["nash_conv"])

    def test_bluff_threshold_uses_actual_terminal_payoffs(self):
        root = poker_root(call=0)
        solver = CFRSolver()
        bet = root.children["BET_25"]
        fold = solver.run_cfr(bet.children["FOLD"], HERO, 0, 1, 1, 500)
        called = solver.compute_utility(bet.children["CALL"], 0, 500)
        for fold_frequency, expected in ((.1, -12.5), (.2, 0), (.4, 25)):
            with self.subTest(fold_frequency=fold_frequency):
                self.assertAlmostEqual(fold_frequency * fold + (1-fold_frequency) * called, expected)

    def test_fixed_returns_and_bluff_frequencies_across_iteration_budgets(self):
        cases = scenarios()
        for name, best, advantage in (("known_returns", "B", 4),
                                      ("bluff_insufficient", "CHECK", 12.5),
                                      ("bluff_sufficient", "BET", 25),
                                      ("bluff_break_even", None, 0)):
            for iterations in ITERATIONS:
                with self.subTest(scenario=name, iterations=iterations):
                    root, factory, equity = cases[name]
                    solver = factory()
                    strategy = solver.solve(root, HERO, equity, iterations)
                    self.assertAlmostEqual(math.fsum(strategy.values()), 1, delta=1e-12)
                    if best is not None:
                        self.assertEqual(max(strategy, key=strategy.get), best)
                    # No equilíbrio de blefe ambos têm valor zero; qualquer
                    # distribuição é ótima, não se exige uma mistura arbitrária.
                    self.assertAlmostEqual(full_information_gap(root, solver, equity),
                                           advantage/(2*iterations), delta=1e-12)

    def test_different_tree_sizes_have_finite_strategies_and_best_response_gaps(self):
        sizes = []
        for name in ("river", "turn", "flop"):
            root, factory, equity = scenarios()[name]
            sizes.append(count_nodes(root))
            solver = factory()
            strategy = solver.solve(root, HERO, equity, 30)
            self.assertAlmostEqual(math.fsum(strategy.values()), 1, delta=1e-12)
            gap = full_information_gap(root, solver, equity)
            self.assertTrue(math.isfinite(gap))
            self.assertGreaterEqual(gap, -1e-10)
        self.assertEqual(sizes, sorted(set(sizes)))

    def test_full_tree_visit_count_scales_with_iteration_budget(self):
        class CountingSolver(CFRSolver):
            visits = 0

            def run_cfr(self, *args, **kwargs):
                self.visits += 1
                return super().run_cfr(*args, **kwargs)

        root = poker_root("TURN")
        solver = CountingSolver()
        solver.solve(root, HERO, 40, 3)
        self.assertEqual(solver.visits, 3 * count_nodes(root))


class TestPrecisionIntegration(unittest.TestCase):
    def test_argmax_uses_full_precision_not_a_rounding_tie(self):
        # Ambas seriam 0.500; argmax deve continuar usando a maior real.
        strategy = {"FOLD": .4999, "CALL": .5001}
        with patch("src.engine.cfr_solver.CFRSolver.solve", return_value=strategy):
            result = make_decision(40, 60, 0, 100, 20, 500, state="RIVER",
                                   hero_cards=HERO, board_cards=poker_root().board_cards, equity=40)
        self.assertEqual(result["selected_action"], "CALL")
        self.assertEqual(result["selected_action_probability"], .5001)

    def test_real_solver_precision_reaches_cache_and_only_panel_rounds(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = HERO
        self.app.hero_stack = 500
        self.app.num_opponents = 1
        output = io.StringIO()
        with patch("main.make_decision", wraps=make_decision) as decision, redirect_stdout(output):
            first = self.app.process_cards(poker_root().board_cards, 100, 20, True)
            second = self.app.process_cards(poker_root().board_cards, 100, 20, True)
        probability = first["selected_action_probability"]
        strategy = first["cfr_strategy"]
        self.assertEqual(probability, strategy[first["selected_action"]])
        self.assertAlmostEqual(math.fsum(strategy.values()), 1, delta=1e-12)
        self.assertNotEqual(probability, round(probability, 3))
        self.assertIs(first, second)
        self.assertIn(f"Probabilidade: {probability:.3f}", output.getvalue())
        decision.assert_called_once()
        self.calculate_equity.assert_called_once()
        self.assertFalse(first["action_available"])


if __name__ == "__main__":
    unittest.main()
