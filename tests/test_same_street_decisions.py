"""4A.3: turnos repetidos, identidade da mão e cache probabilístico separado.

Somente captura/OCR, sorteio Monte Carlo e escrita de dataset são fronteiras
controladas. FSM, ranges, validação financeira, risco e CFR+ são reais.
O desaparecimento dos botões não informa qual ação o Hero executou.
"""

import io
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from main import run_live
from src.engine.evaluator import FULL_DECK, calculate_equity
from src.engine.range_model import OpponentModel
from src.engine.risk_engine import make_decision
from src.engine.state_machine import HandState
from tests import test_app_cache


HERO = ["Kc", "4d"]
OTHER = ["Ac", "8c"]
FLOP = ["Qs", "7h", "2d"]
TURN = FLOP + ["Jc"]
RIVER = TURN + ["6s"]


def observation(hero=HERO, board=(), pot=5, call=0, turn=True, stack=1000, actions=None):
    return dict(hero=list(hero), board=list(board), pot=pot, call=call,
                turn=turn, stack=stack, actions=actions)


def context(frame):
    active = frame["turn"]
    call = frame["call"] if active else None
    kind = ("CHECK" if call == 0 else "CALL") if active else "WAITING"
    actions = frame["actions"]
    if actions is None:
        actions = (["FOLD", "CHECK", "RAISE"] if call == 0 else ["FOLD", "CALL", "RAISE"]) if active else []
    return dict(is_hero_turn=active, action_type=kind, bet_to_call=call, available_actions=actions)


class TestSameStreetDecisions(unittest.TestCase):
    def setUp(self):
        test_app_cache.TestPokerAnalyticsAppCache.setUp(self)
        self.app.hero_cards = list(HERO)
        self.app.opponent_model = OpponentModel(FULL_DECK)

    def live(self, frames, debug=False):
        output = io.StringIO()
        with patch("main.PokerAnalyticsApp", return_value=self.app), \
             patch("main.grab_screen", side_effect=frames + [KeyboardInterrupt]), \
             patch("main.find_and_detect_hero_cards", side_effect=lambda f, **kw: (f["hero"], None)), \
             patch("main.detect_board", side_effect=lambda f, **kw: f["board"]), \
             patch("main.extract_pot", side_effect=lambda f, **kw: f["pot"]), \
             patch("main.detect_turn_and_bet_to_call", side_effect=lambda f, **kw: context(f)), \
             patch("main.extract_hero_stack", side_effect=lambda f, **kw: f["stack"]), \
             patch("main.extract_hero_position", return_value="BTN"), \
             patch("main.detect_active_opponents", return_value=2), \
             patch("main.time.sleep"), \
             patch("main.make_decision", wraps=make_decision) as risk, \
             patch("src.engine.blueprint.PreflopBlueprint.sample_action",
                   side_effect=lambda strategy: max(strategy, key=strategy.get)), \
             patch.object(self.app, "render_dashboard", wraps=self.app.render_dashboard) as dashboard, \
             redirect_stdout(output):
            run_live(debug_hands=debug)
        self.analyses = [call.args[0] for call in dashboard.call_args_list]
        self.risk = risk
        return output.getvalue()

    def assert_same_hand(self, board=()):
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.hero_cards, HERO)
        self.assertEqual(self.app.state_machine.board, list(board))

    def test_check_raise_return_is_new_decision_not_new_hand(self):
        output = self.live([observation(), observation(turn=False), observation(pot=25, call=10)])
        self.assert_same_hand()
        first, second = self.analyses
        self.assertEqual(first["action"], "CHECK")
        self.assertTrue(first["action_available"])
        self.assertNotEqual(second["action"], "CHECK")
        self.assertEqual(second["state"], "PRE_FLOP")
        self.assertEqual(second["bet_to_call"], 10)
        self.assertEqual(second["pot_size"], 25)
        self.assertAlmostEqual(second["ev"], .41 * 35 - 10)
        self.assertEqual(second["action"], "FOLD")  # Blueprint Tier 4, sem mudar estratégia.
        self.assertTrue(second["action_available"])
        self.assertNotEqual(first["reason"], second["reason"])
        self.assertEqual(first["equity"], second["equity"])
        self.assertFalse(first["equity_cached"])
        self.assertTrue(second["equity_cached"])
        self.assertIn("DECISÃO ATUALIZADA - EQUITY EM CACHE", output)
        self.assertIn("Monte Carlo Executados:  1", output)
        self.assertEqual(self.risk.call_count, 2)
        self.calculate_equity.assert_called_once()

    def test_two_divergent_hero_reads_on_empty_preflop_board_cannot_reset(self):
        wrong = observation(hero=["Kd", "4d"])
        output = self.live([observation(), wrong, wrong, observation()])
        self.assert_same_hand()
        self.assertNotIn("[RESET]", output)
        self.assertIn("CAPTURA INCERTA", output)
        self.assertTrue(all(a["hero_cards"] == HERO for a in self.analyses))
        self.calculate_equity.assert_called_once()

    def test_single_divergent_hero_read_recovers_without_reset(self):
        self.live([observation(), observation(hero=["Kd", "4d"]), observation(pot=25, call=10)])
        self.assert_same_hand()
        self.assertEqual(self.risk.call_count, 2)
        self.calculate_equity.assert_called_once()

    def test_partial_hero_read_suspends_only_bad_capture(self):
        self.live([observation(), observation(hero=["Kc", "Desconhecida"]), observation()])
        self.assert_same_hand()
        self.assertEqual(len(self.analyses), 2)
        self.calculate_equity.assert_called_once()

    def test_missing_hero_then_raise_is_not_a_redeal(self):
        raised = observation(pot=25, call=10)
        self.live([observation(), observation(hero=[], turn=False), raised, raised])
        self.assert_same_hand()
        self.assertEqual(self.analyses[-1]["bet_to_call"], 10)
        self.assertNotEqual(self.analyses[-1]["action"], "CHECK")
        self.calculate_equity.assert_called_once()

    def test_buttons_return_with_identical_values_refreshes_decision_only(self):
        self.live([observation(), observation(turn=False), observation()])
        self.assert_same_hand()
        self.assertEqual(self.risk.call_count, 2)
        self.assertIsNot(self.analyses[0], self.analyses[1])
        self.calculate_equity.assert_called_once()

    def test_multiple_raises_same_street_preserve_weights(self):
        frames = [observation(pot=25, call=10), observation(turn=False),
                  observation(pot=45, call=20, stack=990), observation(turn=False),
                  observation(pot=85, call=30, stack=970)]
        self.live(frames)
        self.assert_same_hand()
        self.assertEqual([a["bet_to_call"] for a in self.analyses], [10, 20, 30])
        expected = OpponentModel(FULL_DECK)
        expected.apply_action_heuristic(25, "PRE_FLOP")
        self.assertEqual(self.app.opponent_model.weights, expected.weights)
        self.assertEqual(self.risk.call_count, 3)
        self.calculate_equity.assert_called_once()

    def test_call_then_new_raise_is_still_same_preflop_hand(self):
        hero = ["Kh", "Qh"]  # Tier 2: CALL é a ação mais provável enfrentando aposta.
        self.app.hero_cards = list(hero)
        self.live([observation(hero=hero, pot=25, call=10),
                   observation(hero=hero, pot=25, turn=False),
                   observation(hero=hero, pot=65, call=20, stack=990)])
        self.assertEqual(self.app.state_machine.hand_generation, 0)
        self.assertEqual(self.app.hero_cards, hero)
        self.assertEqual([a["action"] for a in self.analyses], ["CALL", "CALL"])
        self.assertEqual([a["recommended_amount"] for a in self.analyses], [10, 20])
        self.assertTrue(all(a["action_available"] for a in self.analyses))
        self.assertEqual(self.risk.call_count, 2)
        self.calculate_equity.assert_called_once()

    def test_pot_call_stack_and_buttons_update_decision_without_reweighting(self):
        self.live([observation(), observation(pot=10), observation(pot=25, call=10),
                   observation(pot=25, call=10, stack=990, actions=["FOLD", "CALL"])])
        self.assert_same_hand()
        self.assertEqual(self.risk.call_count, 4)
        self.calculate_equity.assert_called_once()
        self.assertEqual(self.analyses[-1]["hero_stack"], 990)
        self.assertEqual(self.analyses[-1]["action_context"]["available_actions"], ["FOLD", "CALL"])

    def test_each_postflop_street_allows_multiple_decisions(self):
        frames = []
        for board in (FLOP, TURN, RIVER):
            frames += [observation(board=board, pot=100), observation(board=board, pot=100, turn=False),
                       observation(board=board, pot=140, call=20)]
        self.live(frames)
        self.assert_same_hand(RIVER)
        self.assertEqual([a["state"] for a in self.analyses], ["FLOP", "FLOP", "TURN", "TURN", "RIVER", "RIVER"])
        self.assertEqual(self.risk.call_count, 6)
        self.assertEqual(self.calculate_equity.call_count, 3)
        for a in self.analyses[1::2]:
            self.assertNotEqual(a["action"], "CHECK")
            self.assertIn(a["selected_action"], a["cfr_strategy"])
            self.assertEqual(a["recommended_amount"], a["action_details"]["additional_amount"])

    def test_legitimate_flop_after_repeated_preflop_actions(self):
        self.live([observation(), observation(pot=25, call=10), observation(board=FLOP, pot=45)])
        self.assert_same_hand(FLOP)
        self.assertEqual(self.app.state_machine.current_state, HandState.FLOP)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_new_preflop_hand_after_observed_end_can_have_different_cards(self):
        new = observation(hero=OTHER, pot=5)
        self.live([observation(), observation(hero=[], pot=0, turn=False), new, new])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.hero_cards, OTHER)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_new_preflop_hand_same_cards_and_same_pot_requires_independent_end(self):
        self.live([observation(), observation(hero=[], pot=0, turn=False), observation(), observation()])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.hero_cards, HERO)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_confirmed_preflop_end_can_recover_directly_on_new_flop(self):
        new = observation(hero=OTHER, board=FLOP, pot=5)
        self.live([observation(), observation(hero=[], pot=0, turn=False), new, new])
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertEqual(self.app.hero_cards, OTHER)
        self.assertEqual(self.app.state_machine.board, FLOP)
        expected = OpponentModel(FULL_DECK)
        expected.apply_postflop_board(FLOP, 5)
        self.assertEqual(self.app.opponent_model.weights, expected.weights)
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_unknown_pot_or_buttons_cannot_confirm_preflop_end(self):
        self.app.process_cards([], 5, 0, False)
        self.app.observe_hand(HERO, [], 5, hero_position="BTN")
        waiting = dict(is_hero_turn=False, action_type="WAITING", available_actions=[])
        unknown = dict(waiting, action_type="UNKNOWN")
        for pot, action_context in ((None, waiting), (5, waiting), (0, unknown),
                                    (0, dict(is_hero_turn=True, action_type="CHECK"))):
            with self.subTest(pot=pot, context=action_context):
                self.assertEqual(self.app.observe_hand([], [], pot, action_context=action_context), "UNCERTAIN")
                for _ in range(2):
                    self.assertEqual(self.app.observe_hand(OTHER, [], 15), "UNCERTAIN")
                self.assert_same_hand()
        # Sem posição atual, o intervalo permanece ambíguo; não presume continuidade.
        self.assertEqual(self.app.observe_hand(HERO, [], 5), "UNCERTAIN")
        self.assertEqual(self.app.observe_hand(HERO, [], 5, hero_position="BTN"), "CURRENT")
        self.assert_same_hand()

    def test_different_preflop_cards_without_independent_end_remain_uncertain(self):
        new = observation(hero=OTHER, pot=15)
        self.live([observation(), new, new, new])
        self.assert_same_hand()
        self.assertEqual(len(self.analyses), 1)

    def test_fold_observation_has_no_old_recommendation_until_new_deal(self):
        new = observation(hero=OTHER, pot=5)
        output = self.live([observation(pot=25, call=10), observation(hero=[], pot=25, turn=False),
                            observation(hero=[], pot=0, turn=False), new, new])
        self.assertEqual(len(self.analyses), 2)
        self.assertEqual(self.app.state_machine.hand_generation, 1)
        self.assertIn("CAPTURA INCERTA", output)

    def test_invalid_finances_do_not_replace_cache_or_update_ranges(self):
        self.live([observation(pot=25, call=10), observation(pot=45, call=20, stack=5),
                   observation(pot=45, call=20, stack=990)])
        self.assert_same_hand()
        self.assertFalse(self.analyses[1]["observation_valid"])
        self.assertFalse(self.analyses[1]["action_available"])
        self.assertEqual(self.analyses[1]["financial_observation"]["hero_stack"], 5)
        self.assertEqual(self.risk.call_count, 2)
        self.calculate_equity.assert_called_once()

    def test_iterations_and_opponent_count_invalidate_probability_cache(self):
        with redirect_stdout(io.StringIO()):
            self.app.process_cards([], 25, 10, False)
            self.app.iterations += 1
            self.app.process_cards([], 25, 10, False)
            self.app.num_opponents += 1
            self.app.process_cards([], 25, 10, False)
        self.assertEqual(self.calculate_equity.call_count, 3)

    def test_uncertain_board_does_not_consume_probability_cache(self):
        with redirect_stdout(io.StringIO()):
            first = self.app.process_cards(FLOP, 100, 10, False)
            bad = self.app.process_cards(["Desconhecida"] * 5, 100, 10, False)
            recovered = self.app.process_cards(FLOP, 140, 20, False)
        self.assertFalse(bad["observation_valid"])
        self.assertEqual(recovered["equity"], first["equity"])
        self.calculate_equity.assert_called_once()

    def test_different_range_model_with_same_revision_invalidates_cache(self):
        with redirect_stdout(io.StringIO()):
            first = self.app.process_cards(FLOP, 100, 10, False)
            replacement = OpponentModel(FULL_DECK)
            replacement.apply_postflop_board(FLOP, 100)
            self.assertEqual(replacement.revision, self.app.opponent_model.revision)
            self.app.opponent_model = replacement
            second = self.app.process_cards(FLOP, 100, 10, False)
        self.assertIsNot(first, second)
        self.assertFalse(second["equity_cached"])
        self.assertEqual(self.calculate_equity.call_count, 2)

    def test_real_seeded_monte_carlo_is_reused_but_ev_is_updated(self):
        previous_random_state = random.getstate()
        try:
            random.seed(20261011)
            with patch("main.calculate_equity", wraps=calculate_equity) as simulation, \
                 patch("main.make_decision", wraps=make_decision) as risk, \
                 redirect_stdout(io.StringIO()):
                first = self.app.process_cards(FLOP, 100, 0, False)
                weights = dict(self.app.opponent_model.weights)
                for pot in (140, 150, 140):
                    second = self.app.process_cards(FLOP, pot, 20, False)
                    self.assertEqual(second["equity"], first["equity"])
                    self.assertAlmostEqual(second["ev"], first["equity"] / 100 * (pot + 20) - 20)
                    self.assertEqual(self.app.opponent_model.weights, weights)
                simulation.assert_called_once()
                self.assertEqual(risk.call_count, 4)
        finally:
            random.setstate(previous_random_state)


if __name__ == "__main__":
    unittest.main()
