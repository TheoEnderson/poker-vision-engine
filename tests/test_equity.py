"""Participação no pote em empates: mãos reais e sorteios controlados."""

import io
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from src.engine.evaluator import FULL_DECK, calculate_equity, evaluate_7_cards
from src.engine.range_model import OpponentModel
from src.engine.risk_engine import calculate_ev, make_decision
from tests import test_app_cache


BOARD = ["2c", "3d", "7h", "9s", "Jc"]
HERO = ["As", "Kd"]
TIED = [["Ah", "Kh"], ["Ad", "Kc"], ["Ac", "Ks"]]
WEAKER = [["Qh", "10h"], ["Qd", "10c"], ["Qc", "10s"]]
ROYAL = ["10s", "Js", "Qs", "Ks", "As"]


class TestExactEquity(unittest.TestCase):
    def simulate(self, hero, board, deals):
        """Somente o sorteio é controlado; avaliação e agregação são reais."""
        samples = []
        for opponents in deals:
            cards = hero + board + [c for hand in opponents for c in hand]
            self.assertEqual(len(cards), len(set(cards)))
            samples.append([])  # Board completo: nenhuma carta a sortear.
            hero_score = evaluate_7_cards(hero + board)
            for hand in opponents:
                samples.append(hand)
                # A simulação pode encerrar quando a derrota já é certa.
                if evaluate_7_cards(hand + board) > hero_score:
                    break

        def draw(population, k):
            cards = samples.pop(0)
            self.assertEqual(len(cards), k)
            self.assertTrue(set(cards) <= set(population))
            return cards

        with patch("src.engine.evaluator.random.sample", side_effect=draw):
            result = calculate_equity(hero, board, len(deals[0]), len(deals))
        self.assertEqual(samples, [])
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(sum(result[:3]) / 100, 1, delta=0.0002)
        self.assertTrue(0 <= result[3] / 100 <= 1)
        return result[:4]

    def test_exclusive_win(self):
        self.assertEqual(self.simulate(HERO, BOARD, [WEAKER]), (100, 0, 0, 100))

    def test_exclusive_loss(self):
        self.assertEqual(self.simulate(HERO, BOARD, [[["2h", "2d"]]]), (0, 0, 100, 0))

    def test_hero_ties_with_one_two_and_three_opponents(self):
        for count in (1, 2, 3):
            with self.subTest(opponents=count):
                self.assertEqual(self.simulate(HERO, BOARD, [TIED[:count]]),
                                 (0, 100, 0, round(100 / (count + 1), 2)))

    def test_community_royal_for_every_supported_opponent_count(self):
        hero = ["2h", "3h"]
        remaining = [c for c in FULL_DECK if c not in hero + ROYAL]
        opponents = [remaining[i:i + 2] for i in range(0, 16, 2)]
        for count in range(1, 9):
            with self.subTest(players=count + 1):
                self.assertEqual(self.simulate(hero, ROYAL, [opponents[:count]]),
                                 (0, 100, 0, round(100 / (count + 1), 2)))

    def test_opponents_tie_but_hero_loses(self):
        hero = WEAKER[0]
        self.assertEqual(evaluate_7_cards(TIED[0] + BOARD), evaluate_7_cards(TIED[1] + BOARD))
        self.assertEqual(self.simulate(hero, BOARD, [TIED[:2]]), (0, 0, 100, 0))

    def test_only_some_opponents_share_the_best_hand(self):
        for opponents in ([TIED[0], *WEAKER[:2]], [*TIED[:2], WEAKER[0]]):
            with self.subTest(opponents=opponents):
                winners = 1 + sum(hand in TIED for hand in opponents)
                self.assertEqual(self.simulate(HERO, BOARD, [opponents]),
                                 (0, 100, 0, round(100 / winners, 2)))

    def test_tie_with_one_opponent_is_overridden_by_a_better_hand(self):
        self.assertEqual(self.simulate(HERO, BOARD, [[TIED[0], ["2h", "2d"]]]),
                         (0, 0, 100, 0))

    def test_mixed_winner_counts_preserve_outcome_frequencies(self):
        deals = [WEAKER, [TIED[0], *WEAKER[:2]], [*TIED[:2], WEAKER[0]],
                 TIED, [*TIED[:2], ["2h", "2d"]]]
        # Uma vitória, três empates (2/3/4 vencedores) e uma derrota.
        expected = round(100 * (1 + 1/2 + 1/3 + 1/4) / 5, 2)
        result = self.simulate(HERO, BOARD, deals)
        self.assertEqual(result, (20, 60, 20, expected))
        self.assertNotEqual(result[3], result[0] + result[1] / 2)


class TestEquityMonteCarlo(unittest.TestCase):
    def test_seeded_weighted_ranges_and_collision_free_deals(self):
        model = OpponentModel(FULL_DECK)
        flop = BOARD[:3]
        model.apply_postflop_board(flop, 100)
        weights = dict(model.weights)
        previous = random.getstate()
        used = set()

        def checked_evaluation(cards):
            self.assertEqual(len(cards), len(set(cards)))
            if cards[:2] == HERO:
                used.clear()
                used.update(cards)
            else:
                self.assertFalse(set(cards[:2]) & used)
                used.update(cards[:2])
            return evaluate_7_cards(cards)

        try:
            results = []
            for _ in range(2):
                random.seed(20261011)
                with patch("src.engine.evaluator.evaluate_7_cards", side_effect=checked_evaluation):
                    results.append(calculate_equity(HERO, flop, 3, 500, model)[:4])
            self.assertEqual(results[0], results[1])
            self.assertAlmostEqual(sum(results[0][:3]), 100, delta=0.02)
            self.assertTrue(0 <= results[0][3] <= 100)
            self.assertEqual(model.weights, weights)
        finally:
            random.setstate(previous)

    def test_weighted_royal_equity_is_exact_and_ranges_remain_unchanged(self):
        model = OpponentModel(FULL_DECK)
        model.apply_postflop_board(ROYAL, 100)
        weights = dict(model.weights)
        previous = random.getstate()
        try:
            random.seed(20261011)
            self.assertEqual(calculate_equity(["2h", "3h"], ROYAL, 3, 100, model)[:4],
                             (0, 100, 0, 25))
            self.assertEqual(model.weights, weights)
        finally:
            random.setstate(previous)


class TestEquityConsumers(unittest.TestCase):
    def setUp(self):
        previous = random.getstate()
        self.addCleanup(random.setstate, previous)
        random.seed(20261011)

    def test_risk_engine_accepts_explicit_equity_for_ev(self):
        for equity in (0, 25):
            with self.subTest(equity=equity):
                result = make_decision(0, 0, 100, 120, 20, 1000, equity=equity)
                self.assertEqual(result["equity"], equity)
                self.assertEqual(result["ev"], calculate_ev(0, 0, 100, 120, 20, equity=equity))

    def test_legacy_risk_call_remains_compatible(self):
        result = make_decision(40, 58, 2, 120, 20, 1000, num_opponents=1)
        self.assertEqual(result["equity"], 41)

    def test_correct_equity_reaches_solver_cache_and_dashboard(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = ["2h", "3h"]
        self.app.num_opponents = 3
        self.app.opponent_model = OpponentModel(FULL_DECK)
        output = io.StringIO()
        # App/FSM/ranges/Monte Carlo/motor/painel reais; apenas a estratégia é
        # controlada, pois o contrato aqui é o valor recebido pelo solver.
        with patch("main.calculate_equity", wraps=calculate_equity) as simulation, \
             patch("main.make_decision", wraps=make_decision), \
             patch("src.engine.cfr_solver.CFRSolver.solve", return_value={"CALL": 1}) as solve, \
             redirect_stdout(output):
            first = self.app.process_cards(ROYAL, 120, 20, True)
            second = self.app.process_cards(ROYAL, 120, 20, True)
        self.assertEqual(first["equity"], 25)
        self.assertEqual(first["p_tie"], 100)
        self.assertEqual(solve.call_args.args[2], 25)
        self.assertEqual(first["ev"], 15)  # 25% de (120 + 20), menos o call de 20.
        self.assertIs(first, second)
        simulation.assert_called_once()
        self.assertRegex(output.getvalue(), r"Equity Consolidada:\s+25\.00%")
        self.assertEqual(self.app.dataset_logger.log_state.call_args.args[0]["equity"], 25)


if __name__ == "__main__":
    unittest.main()
