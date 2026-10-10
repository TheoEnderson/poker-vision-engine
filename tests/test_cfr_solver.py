import unittest
from src.engine.game_tree import GameNode, GameTreeBuilder
from src.engine.cfr_solver import CFRSolver

class TestCFRSolver(unittest.TestCase):
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
        
        self.assertIsNotNone(strategy)
        self.assertIn("FOLD", strategy)
        self.assertIn("CALL", strategy)
        # A soma das probabilidades deve ser próxima de 1
        self.assertTrue(abs(sum(strategy.values()) - 1.0) < 0.01)

if __name__ == '__main__':
    unittest.main()
