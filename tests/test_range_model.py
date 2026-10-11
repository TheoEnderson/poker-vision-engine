"""Evidências de ranges: idempotência, normalização e novas streets.

Exemplo matemático reproduzível:
    venv/bin/python -B -m tests.test_range_model --example
"""

import math
import random
import sys
import unittest

from src.engine.evaluator import FULL_DECK, calculate_equity
from src.engine.range_model import OpponentModel


HERO = ["8h", "8d"]
FLOP = ["8s", "4c", "Jd"]
TURN = FLOP + ["6d"]
RIVER = TURN + ["Ac"]
STRONG = ("As", "Ac")
WEAK = ("2h", "6s")
SEED = 20261011


def seeded_equity(model, board, iterations=500):
    previous = random.getstate()
    try:
        random.seed(SEED)
        # Tempo de execução não faz parte da comparação matemática.
        return calculate_equity(HERO, board, num_opponents=3, iterations=iterations,
                                opp_range_percent=model)[:4]
    finally:
        random.setstate(previous)


class TestRangeEvidence(unittest.TestCase):
    def setUp(self):
        self.model = OpponentModel(FULL_DECK)

    def assert_distribution(self, weights):
        self.assertTrue(weights)
        self.assertTrue(all(math.isfinite(w) and w > 0 for w in weights))
        self.assertAlmostEqual(math.fsum(weights), 1.0, places=12)

    def test_initial_chen_prior_is_normalized_without_changing_ratios(self):
        self.assertEqual(len(self.model.weights), 1326)
        self.assert_distribution(self.model.weights.values())
        self.assertAlmostEqual(self.model.weights[STRONG] / self.model.weights[WEAK], 22.0)

    def test_first_board_keeps_existing_likelihood_factors(self):
        self.model.apply_postflop_board(FLOP, 318)
        self.assertAlmostEqual(self.model.weights[STRONG] / self.model.weights[WEAK], 66.0)
        self.assert_distribution(self.model.weights.values())

    def test_second_identical_board_does_not_change_any_weight(self):
        self.model.apply_postflop_board(FLOP, 318)
        first = dict(self.model.weights)
        self.model.apply_postflop_board(FLOP, 318)
        self.assertEqual(self.model.weights, first)

    def test_ten_reapplications_with_oscillating_pots_are_identical(self):
        self.model.apply_postflop_board(FLOP, 318)
        first = dict(self.model.weights)
        for pot in [320, 318, 295, 320, 29, 31, 318, 400, 315, 318]:
            self.model.apply_postflop_board(FLOP, pot)
            self.assertEqual(self.model.weights, first)

    def test_crossing_legacy_pot_threshold_is_not_a_new_board(self):
        prior = dict(self.model.weights)
        self.model.apply_postflop_board(FLOP, 29)
        self.assertEqual(self.model.weights, prior)
        self.model.apply_postflop_board(FLOP, 31)
        self.assertEqual(self.model.weights, prior)
        self.model.apply_postflop_board(TURN, 31)
        self.assertNotEqual(self.model.weights, prior)

    def test_board_order_and_card_aliases_do_not_duplicate_evidence(self):
        self.model.apply_postflop_board(["10s", "4c", "Jd"], 100)
        first = dict(self.model.weights)
        self.model.apply_postflop_board(["JD", "Ts", "4c"], 110)
        self.assertEqual(self.model.weights, first)

    def test_flop_turn_river_update_once_and_retain_prior_information(self):
        self.model.apply_action_heuristic(100, "PRE_FLOP")
        for board in (FLOP, TURN, RIVER):
            before = dict(self.model.weights)
            self.model.apply_postflop_board(board, 318)
            self.assertNotEqual(self.model.weights, before)
            self.assert_distribution(self.model.weights.values())
            after = dict(self.model.weights)
            self.model.apply_postflop_board(board, 320)
            self.assertEqual(self.model.weights, after)
        # Um modelo iniciado no river não possui a informação prévia acumulada.
        river_only = OpponentModel(FULL_DECK)
        river_only.apply_postflop_board(RIVER, 318)
        self.assertNotEqual(self.model.weights, river_only.weights)
        self.model.apply_postflop_board(FLOP, 318)
        self.assertEqual(self.model.weights, after)

    def test_turn_and_river_apply_existing_factors_to_previous_distribution(self):
        self.model.apply_postflop_board(FLOP, 318)
        newly_paired, still_missed = ("6h", "10s"), ("2s", "3h")
        old_ratio = self.model.weights[newly_paired] / self.model.weights[still_missed]
        self.model.apply_postflop_board(TURN, 318)
        self.assertAlmostEqual(self.model.weights[newly_paired] / self.model.weights[still_missed], old_ratio * 3)
        queens, aces = ("Qs", "Qh"), ("Ah", "Ad")
        old_ratio = self.model.weights[queens] / self.model.weights[aces]
        self.model.apply_postflop_board(RIVER, 318)
        self.assertAlmostEqual(self.model.weights[queens] / self.model.weights[aces], old_ratio / 3)

    def test_pot_heuristic_is_one_initial_context_not_an_action_stream(self):
        self.model.apply_action_heuristic(25, "PRE_FLOP")
        first = dict(self.model.weights)
        for pot in [25, 100, 15, 320] * 3:
            self.model.apply_action_heuristic(pot, "PRE_FLOP")
            self.assertEqual(self.model.weights, first)

    def test_legacy_postflop_action_context_is_also_idempotent(self):
        self.model.apply_action_heuristic(100, "FLOP")
        first = dict(self.model.weights)
        self.model.apply_action_heuristic(200, "FLOP")
        self.assertEqual(self.model.weights, first)

    def test_reset_restores_prior_and_allows_same_evidence_again(self):
        initial = dict(self.model.weights)
        self.model.apply_action_heuristic(100, "PRE_FLOP")
        self.model.apply_postflop_board(FLOP, 318)
        first_hand = dict(self.model.weights)
        self.model.reset()
        self.assertEqual(self.model.weights, initial)
        self.model.apply_action_heuristic(100, "PRE_FLOP")
        self.model.apply_postflop_board(FLOP, 318)
        self.assertEqual(self.model.weights, first_hand)

    def test_blockers_return_normalized_distribution_without_pruning_small_weights(self):
        self.model.apply_postflop_board(FLOP, 318)
        before = dict(self.model.weights)
        dead = set(HERO + FLOP)
        combos, weights = self.model.get_weighted_combos(dead)
        self.assertEqual(len(combos), math.comb(47, 2))
        self.assertTrue(all(not (set(combo) & dead) for combo in combos))
        self.assert_distribution(weights)
        self.assertAlmostEqual(weights[combos.index(STRONG)] / weights[combos.index(WEAK)], 66)
        self.assertEqual(self.model.weights, before)

    def test_all_blocked_combos_return_empty_distribution(self):
        self.assertEqual(self.model.get_weighted_combos(set(FULL_DECK)), ([], []))


class TestRangeMonteCarlo(unittest.TestCase):
    def test_fixed_seed_equity_is_identical_after_ten_reapplications(self):
        model = OpponentModel(FULL_DECK)
        model.apply_postflop_board(FLOP, 318)
        weights = dict(model.weights)
        before = seeded_equity(model, FLOP)
        for _ in range(10):
            model.apply_postflop_board(FLOP, 320)
        self.assertEqual(model.weights, weights)
        self.assertEqual(seeded_equity(model, FLOP), before)
        self.assertAlmostEqual(sum(before[:3]), 100.0, places=2)

    def test_same_evidence_sequence_and_seed_are_reproducible(self):
        models = [OpponentModel(FULL_DECK), OpponentModel(FULL_DECK)]
        for model in models:
            model.apply_action_heuristic(25, "PRE_FLOP")
            for board in (FLOP, TURN, RIVER):
                model.apply_postflop_board(board, 318)
        self.assertEqual(models[0].weights, models[1].weights)
        self.assertEqual(seeded_equity(models[0], RIVER), seeded_equity(models[1], RIVER))


def print_example():
    model = OpponentModel(FULL_DECK)
    def report(label, board):
        equity = seeded_equity(model, board, iterations=10000)
        print(f"{label}: soma={math.fsum(model.weights.values()):.12f}; "
              f"P(AsAc)={model.weights[STRONG]:.10f}; P(2h6s)={model.weights[WEAK]:.10f}; "
              f"razão={model.weights[STRONG] / model.weights[WEAK]:.6f}; "
              f"P_win/P_tie/P_lose/equity={equity}")
    print(f"Hero={HERO}; flop={FLOP}; turn={TURN}; adversários=3; iterações=10000; seed={SEED}")
    report("Prior sem evidência de board", FLOP)
    model.apply_postflop_board(FLOP, 318)
    report("Primeira evidência: flop", FLOP)
    first = dict(model.weights)
    for _ in range(10):
        model.apply_postflop_board(FLOP, 320)
    report("Dez repetições do flop, pote 320", FLOP)
    print(f"Pesos exatamente iguais após repetições: {model.weights == first}")
    model.apply_postflop_board(TURN, 320)
    report("Nova evidência: turn 6d", TURN)


if __name__ == "__main__":
    if "--example" in sys.argv:
        print_example()
    else:
        unittest.main()
