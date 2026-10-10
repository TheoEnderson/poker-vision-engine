"""
src/engine/game_tree.py
Representação da Árvore de Jogo (Game Tree) e Abstração de Ações para o Solver CFR.
"""

from typing import List, Dict, Optional, Any
import hashlib

# Abstração de Ações
# Limitamos as ações a FOLD, CALL (inclui CHECK) e Raises proporcionais ao pote.
ACTION_FOLD = "FOLD"
ACTION_CALL = "CALL" # Serve como CHECK se não houver aposta anterior
ACTION_BET_25 = "BET_25"
ACTION_BET_50 = "BET_50"
ACTION_BET_100 = "BET_100"
ACTION_ALL_IN = "ALL_IN"

VALID_ACTIONS = [ACTION_FOLD, ACTION_CALL, ACTION_BET_25, ACTION_BET_50, ACTION_BET_100, ACTION_ALL_IN]

class GameNode:
    """
    Representa um Nó na Árvore de Decisão.
    """
    def __init__(
        self, 
        player: int, # 0 = Hero, 1 = Villain, -1 = Chance/Nature (Deal cards)
        pot_size: float,
        hero_stack: float,
        villain_stack: float,
        board_cards: List[str],
        history: str = "",
        current_street: str = "PRE_FLOP",
        bet_to_call: float = 0.0
    ):
        self.player = player
        self.pot_size = pot_size
        self.hero_stack = hero_stack
        self.villain_stack = villain_stack
        self.board_cards = board_cards
        self.history = history
        self.current_street = current_street
        self.bet_to_call = bet_to_call
        
        self.children: Dict[str, 'GameNode'] = {}
        
    def is_terminal(self) -> bool:
        """Verifica se o nó é terminal (Fim da mão)."""
        if self.history and self.history.endswith("FOLD"):
            return True
        if self.hero_stack == 0 and self.villain_stack == 0 and self.current_street == "SHOWDOWN":
            return True
        if self.current_street == "SHOWDOWN":
            return True
        return False

    def get_info_set(self, hero_cards: List[str]) -> str:
        """
        Retorna a string do Information Set (I) para o jogador atual.
        Ex: "AhKs|QhJh2c|CALL.BET_50"
        """
        cards_str = "".join(sorted(hero_cards))
        board_str = "".join(sorted(self.board_cards))
        return f"{cards_str}|{board_str}|{self.history}"

class GameTreeBuilder:
    """
    Constrói a árvore de ações recursivamente a partir de um estado atual.
    """
    def __init__(self, max_depth_per_street: int = 3):
        self.max_depth_per_street = max_depth_per_street

    def build_tree(self, root: GameNode, current_depth: int = 0) -> GameNode:
        if root.is_terminal():
            return root
            
        actions_to_explore = self._get_valid_actions(root, current_depth)
        
        for action in actions_to_explore:
            child = self._apply_action(root, action)
            next_depth = current_depth + 1 if action not in [ACTION_FOLD, ACTION_CALL] else 0
            
            # Avança a rua se:
            # 1. Alguém deu CALL numa aposta.
            # 2. Ambos deram CHECK (ação CALL quando bet_to_call == 0 e a história desta street tem pelo menos uma ação).
            is_check = (action == ACTION_CALL and root.bet_to_call == 0)
            is_call = (action == ACTION_CALL and root.bet_to_call > 0)
            
            # Conta as ações desde o último avanço de street
            street_hist = root.history.split("|")[-1] if root.history else ""
            actions_this_street = len([a for a in street_hist.split(".") if a])
            
            if is_call or (is_check and actions_this_street >= 1):
                child = self._advance_street(child)
                next_depth = 0
                
            root.children[action] = self.build_tree(child, next_depth)
            
        return root

    def _get_valid_actions(self, node: GameNode, current_depth: int) -> List[str]:
        actions = [ACTION_FOLD, ACTION_CALL]
        
        if current_depth < self.max_depth_per_street:
            player_stack = node.hero_stack if node.player == 0 else node.villain_stack
            if player_stack > node.bet_to_call:
                if player_stack >= (node.pot_size * 0.25): actions.append(ACTION_BET_25)
                if player_stack >= (node.pot_size * 0.50): actions.append(ACTION_BET_50)
                if player_stack >= node.pot_size: actions.append(ACTION_BET_100)
                actions.append(ACTION_ALL_IN)
                
        return actions

    def _apply_action(self, node: GameNode, action: str) -> GameNode:
        next_player = 1 if node.player == 0 else 0
        new_pot = node.pot_size
        new_hero_stack = node.hero_stack
        new_villain_stack = node.villain_stack
        new_bet_to_call = 0.0
        
        player_stack = node.hero_stack if node.player == 0 else node.villain_stack
        
        if action == ACTION_FOLD:
            pass 
        elif action == ACTION_CALL:
            call_amount = min(player_stack, node.bet_to_call)
            new_pot += call_amount
            if node.player == 0:
                new_hero_stack -= call_amount
            else:
                new_villain_stack -= call_amount
        elif action.startswith("BET_"):
            pct = float(action.split("_")[1]) / 100.0
            bet_amount = node.pot_size * pct
            bet_amount = min(player_stack, bet_amount)
            new_pot += bet_amount
            new_bet_to_call = bet_amount
            if node.player == 0:
                new_hero_stack -= bet_amount
            else:
                new_villain_stack -= bet_amount
        elif action == ACTION_ALL_IN:
            bet_amount = player_stack
            new_pot += bet_amount
            new_bet_to_call = bet_amount
            if node.player == 0:
                new_hero_stack -= bet_amount
            else:
                new_villain_stack -= bet_amount

        hist = node.history + "." + action if node.history else action

        return GameNode(
            player=next_player,
            pot_size=new_pot,
            hero_stack=new_hero_stack,
            villain_stack=new_villain_stack,
            board_cards=node.board_cards,
            history=hist,
            current_street=node.current_street,
            bet_to_call=new_bet_to_call
        )

    def _advance_street(self, node: GameNode) -> GameNode:
        streets = ["PRE_FLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        try:
            idx = streets.index(node.current_street)
            next_street = streets[idx + 1] if idx + 1 < len(streets) else "SHOWDOWN"
        except ValueError:
            next_street = "SHOWDOWN"
            
        hist = node.history + "|" if node.history else "|"
        
        return GameNode(
            player=0, 
            pot_size=node.pot_size,
            hero_stack=node.hero_stack,
            villain_stack=node.villain_stack,
            board_cards=node.board_cards,
            history=hist,
            current_street=next_street,
            bet_to_call=0.0
        )
