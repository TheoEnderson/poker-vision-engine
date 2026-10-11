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
        
        # Sem aposta, CALL representa CHECK; FOLD não pertence a este nó.
        self.assertNotIn("FOLD", tree.children)
        self.assertEqual(set(tree.children), {"CALL", "BET_25", "BET_50", "BET_100", "ALL_IN"})
        
        # Se Hero CALL(CHECK), a vez passa para o Villain
        call_node = tree.children["CALL"]
        self.assertEqual(call_node.player, 1)
        self.assertEqual(call_node.current_street, "FLOP")
        self.assertEqual(call_node.pot_size, 100.0)
        self.assertEqual(call_node.hero_stack, 500.0)
        self.assertEqual(call_node.villain_stack, 500.0)
        self.assertEqual(call_node.bet_to_call, 0.0)
        self.assertNotIn("FOLD", call_node.children)

        # Dois CHECKs encerram a rodada sem movimentar fichas.
        checked_turn = call_node.children["CALL"]
        self.assertEqual(checked_turn.current_street, "TURN")
        self.assertEqual(checked_turn.player, 0)
        self.assertEqual(checked_turn.pot_size, 100.0)
        self.assertEqual(checked_turn.hero_stack, 500.0)
        self.assertEqual(checked_turn.villain_stack, 500.0)
        self.assertEqual(checked_turn.bet_to_call, 0.0)
        
        # Se Villain BET_50, Hero deve responder
        self.assertIn("BET_50", call_node.children)
        villain_bet_node = call_node.children["BET_50"]
        self.assertEqual(villain_bet_node.player, 0)
        self.assertEqual(villain_bet_node.pot_size, 150.0)
        self.assertEqual(villain_bet_node.bet_to_call, 50.0)
        self.assertEqual(villain_bet_node.hero_stack, 500.0)
        self.assertEqual(villain_bet_node.villain_stack, 450.0)
        self.assertIn("FOLD", villain_bet_node.children)
        self.assertTrue(villain_bet_node.children["FOLD"].is_terminal())

        # Se Hero CALL na aposta de 50, paga fichas e a rua avança.
        hero_call_node = villain_bet_node.children["CALL"]
        self.assertEqual(hero_call_node.current_street, "TURN")
        self.assertEqual(hero_call_node.pot_size, 200.0)
        self.assertEqual(hero_call_node.hero_stack, 450.0)
        self.assertEqual(hero_call_node.villain_stack, 450.0)

    def test_build_tree_facing_bet_includes_terminal_fold(self):
        root = GameNode(
            player=0,
            pot_size=100.0,
            hero_stack=500.0,
            villain_stack=500.0,
            board_cards=["Ah", "Ks", "2d"],
            current_street="FLOP",
            bet_to_call=50.0,
        )
        tree = GameTreeBuilder(max_depth_per_street=2).build_tree(root)
        self.assertIn("FOLD", tree.children)
        self.assertIn("CALL", tree.children)
        fold_node = tree.children["FOLD"]
        self.assertTrue(fold_node.is_terminal())
        self.assertEqual(fold_node.pot_size, 100.0)
        self.assertEqual(fold_node.hero_stack, 500.0)
        self.assertEqual(fold_node.villain_stack, 500.0)

        paid_call = tree.children["CALL"]
        self.assertEqual(paid_call.current_street, "TURN")
        self.assertEqual(paid_call.pot_size, 150.0)
        self.assertEqual(paid_call.hero_stack, 450.0)
        self.assertEqual(paid_call.villain_stack, 500.0)
        self.assertEqual(paid_call.bet_to_call, 0.0)

if __name__ == '__main__':
    unittest.main()

