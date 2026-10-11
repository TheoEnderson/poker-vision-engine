"""Testes de caracterização do cache do orquestrador principal."""

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from main import PokerAnalyticsApp, run_live
from src.engine.state_machine import HandState, PokerHandStateMachine
from tests import test_opponent_detection


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


class TestPokerAnalyticsAppCache(unittest.TestCase):
    def setUp(self):
        # Constrói somente o estado necessário para process_cards(), evitando que
        # __init__ crie diretórios ou inicialize o logger baseado em arquivo.
        self.app = PokerAnalyticsApp.__new__(PokerAnalyticsApp)
        self.app.hero_position = "BTN"
        self.app.state_machine = PokerHandStateMachine(
            initial_state=HandState.WAITING_HAND,
            allow_initial_sync=True,
        )
        self.app.dataset_logger = Mock()
        self.app.hero_cards = ["Ah", "Kh"]
        self.app.hero_stack = 1000.0
        self.app.num_opponents = 2
        self.app.iterations = 100
        self.app.auto_detect_hero = False
        self.app.auto_detect_pot = False
        self.app.auto_detect_turn = False
        self.app.last_board = []
        self.app.last_state = HandState.WAITING_HAND
        self.app.last_analysis = None
        self.app.total_frames_processed = 0
        self.app.total_recalculations = 0
        self.app._new_hand_hero_cards_set = False
        self.app.opponent_model = Mock()

        equity_patcher = patch(
            "main.calculate_equity",
            return_value=(40.0, 2.0, 58.0, 41.0, 0.001),
        )
        draws_patcher = patch("main.detect_draws", return_value=(0, ""))
        decision_patcher = patch(
            "main.make_decision",
            return_value={
                "action": "CALL",
                "ev": 1.5,
                "pot_odds": 28.57,
                "equity": 41.0,
                "recommended_amount": 10.0,
                "reason": "Decisão controlada para teste",
            },
        )

        self.calculate_equity = equity_patcher.start()
        draws_patcher.start()
        decision_patcher.start()
        self.addCleanup(patch.stopall)

    def process(self, *, show_dashboard=False):
        return self.app.process_cards(
            detected_cards=[],
            pot_size=25.0,
            bet_to_call=10.0,
            show_dashboard=show_dashboard,
        )

    def test_same_relevant_data_reuses_cached_analysis(self):
        first = self.process()
        second = self.process()

        self.assertIs(second, first)
        self.assertEqual(self.calculate_equity.call_count, 1)
        self.assertEqual(self.app.total_recalculations, 1)

    def test_stack_change_invalidates_cache(self):
        first = self.process()
        self.app.hero_stack = 900.0

        second = self.process()

        self.assertIsNot(second, first)
        self.assertEqual(second["hero_stack"], 900.0)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.app.total_recalculations, 2)

    def test_opponent_count_change_invalidates_cache(self):
        first = self.process()
        self.app.num_opponents = 4

        second = self.process()

        self.assertIsNot(second, first)
        self.assertEqual(second["num_opponents"], 4)
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.calculate_equity.call_args.kwargs["num_opponents"], 4)
        self.assertEqual(self.app.total_recalculations, 2)

    def test_position_change_preserves_current_cache_invalidation(self):
        first = self.process()
        self.app.hero_position = "CO"

        second = self.process()

        self.assertIsNot(second, first)
        self.assertEqual(second["hero_position"], "CO")
        self.assertEqual(self.calculate_equity.call_count, 2)
        self.assertEqual(self.app.total_recalculations, 2)

    def test_dashboard_uses_values_from_the_rendered_analysis(self):
        analysis = self.process()
        self.app.hero_position = "UTG"
        self.app.num_opponents = 5

        output = io.StringIO()
        with redirect_stdout(output):
            self.app.render_dashboard(analysis, cached=True)

        rendered = ANSI_ESCAPE.sub("", output.getvalue())
        self.assertIn("POSIÇÃO HERO:     BTN", rendered)
        self.assertIn("OPONENTES:       2", rendered)
        self.assertIn("Stack do Hero:       1000.0", rendered)

    def test_live_count_reaches_monte_carlo_and_dashboard_across_streets(self):
        self.app.auto_detect_hero = True
        test_opponent_detection.TestOpponentDetection.setUpClass()
        fixture = test_opponent_detection.TestOpponentDetection()
        counts = [4, 3, 2, 2, 1]
        frames = [fixture.table(range(count)) for count in counts]
        boards = [
            [],
            ["2c", "3d", "7s"],
            ["2c", "3d", "7s", "9c"],
            ["2c", "3d", "7s", "9c", "Jd"],
            ["2c", "3d", "7s", "9c", "Jd"],
        ]
        hero_box = {"left": 1208, "top": 652, "width": 179, "height": 61}
        # Captura, OCR, relógio e análise probabilística são fronteiras controladas;
        # contagem visual, ciclo ao vivo, FSM e invalidação de cache são reais.
        with patch("main.PokerAnalyticsApp", return_value=self.app), \
                patch("main.grab_screen", side_effect=frames + [KeyboardInterrupt]), \
                patch("main.find_and_detect_hero_cards", return_value=(self.app.hero_cards, hero_box)), \
                patch("main.detect_turn_and_bet_to_call", return_value={"is_hero_turn": True, "bet_to_call": 10.0}), \
                patch("main.detect_board", side_effect=boards), \
                patch("main.extract_pot", return_value=25.0), \
                patch("main.extract_hero_position", return_value="BTN"), \
                patch("main.extract_hero_stack", return_value=1000.0), \
                patch("main.time.sleep"), \
                patch.object(self.app, "render_dashboard") as dashboard, \
                redirect_stdout(io.StringIO()):
            run_live()

        self.assertEqual([call.kwargs["num_opponents"] for call in self.calculate_equity.call_args_list], counts)
        analyses = [call.args[0] for call in dashboard.call_args_list]
        self.assertEqual([analysis["num_opponents"] for analysis in analyses], counts)
        self.assertEqual([analysis["state"] for analysis in analyses], ["PRE_FLOP", "FLOP", "TURN", "RIVER", "RIVER"])
        # Os dois últimos frames só diferem na contagem: precisam recalcular.
        self.assertEqual(self.app.total_recalculations, 5)
        self.assertEqual(self.app.last_analysis["num_opponents"], 1)


if __name__ == "__main__":
    unittest.main()
