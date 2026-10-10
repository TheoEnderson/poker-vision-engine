"""
tests/test_game_tree.py
Testes unitários para a Árvore de Jogo e Abstrações de Ação.
"""

import unittest
from src.engine.game_tree import GameNode, GameTreeBuilder

class TestGameTree(unittest.TestCase):
    def test_build_tree_flop(self):
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
        tree = builder.build_tree(root)
        
        # Ações iniciais permitidas (Hero tem 500 num pote de 100)
        # FOLD, CALL(CHECK), BET_25(25), BET_50(50), BET_100(100), ALL_IN
        self.assertIn("CALL", tree.children)
        self.assertIn("BET_50", tree.children)
        self.assertIn("ALL_IN", tree.children)
        
        # Se Hero FOLD, o jogo acaba
        fold_node = tree.children["FOLD"]
        self.assertTrue(fold_node.is_terminal())
        
        # Se Hero CALL(CHECK), a vez passa para o Villain
        call_node = tree.children["CALL"]
        self.assertEqual(call_node.player, 1)
        self.assertEqual(call_node.current_street, "FLOP")
        
        # Se Villain BET_50, Hero deve responder
        if "BET_50" in call_node.children:
            villain_bet_node = call_node.children["BET_50"]
            self.assertEqual(villain_bet_node.player, 0)
            self.assertEqual(villain_bet_node.pot_size, 150.0)
            self.assertEqual(villain_bet_node.bet_to_call, 50.0)
            
            # Se Hero CALL na aposta de 50, a rua deve avançar
            hero_call_node = villain_bet_node.children["CALL"]
            self.assertEqual(hero_call_node.current_street, "TURN")
            self.assertEqual(hero_call_node.pot_size, 200.0)

if __name__ == '__main__':
    unittest.main()

