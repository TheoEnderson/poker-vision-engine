import math
import unittest
from src.engine.game_tree import GameNode, GameTreeBuilder
from src.engine.cfr_solver import CFRSolver

class TestCFRSolver(unittest.TestCase):
    def assert_valid_strategy(self, strategy, root):
        self.assertIsInstance(strategy, dict)
        self.assertEqual(set(strategy), set(root.children))
        self.assertTrue(strategy)
        for action, probability in strategy.items():
            with self.subTest(action=action):
                self.assertTrue(math.isfinite(probability))
                self.assertGreaterEqual(probability, 0.0)
                self.assertLessEqual(probability, 1.0)
        # solve() arredonda cada probabilidade para três casas decimais.
        self.assertAlmostEqual(sum(strategy.values()), 1.0, delta=len(strategy) * 0.0005 + 1e-12)

    def test_cfr_solver_basic(self):
        root = GameNode(
            player=0,
            pot_size=100.0,
            hero_stack=500.0,
            villain_stack=500.0,
            board_cards=["Ah", "Ks", "2d"],
            current_street="FLOP",
            bet_to_call=0.0
        )
        builder = GameTreeBuilder(max_depth_per_street=2)
        root = builder.build_tree(root)
        
        solver = CFRSolver()
        # Hero tem 80% de equity
        strategy = solver.solve(root, hero_cards=["Ad", "As"], equity=80.0, iterations=100)
        
        self.assert_valid_strategy(strategy, root)
        self.assertNotIn("FOLD", root.children)
        self.assertNotIn("FOLD", strategy)
        self.assertIn("CALL", strategy)

    def test_cfr_solver_facing_bet_includes_fold(self):
        root = GameNode(
            player=0,
            pot_size=100.0,
            hero_stack=500.0,
            villain_stack=500.0,
            board_cards=["Ah", "Ks", "2d"],
            current_street="FLOP",
            bet_to_call=50.0,
        )
        root = GameTreeBuilder(max_depth_per_street=2).build_tree(root)
        strategy = CFRSolver().solve(root, hero_cards=["Ad", "As"], equity=80.0, iterations=100)
        self.assert_valid_strategy(strategy, root)
        self.assertIn("FOLD", root.children)
        self.assertIn("FOLD", strategy)
        self.assertIn("CALL", strategy)

if __name__ == '__main__':
    unittest.main()
