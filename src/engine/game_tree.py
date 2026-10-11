"""
src/engine/game_tree.py
Representação da Árvore de Jogo (Game Tree) e Abstração de Ações para o Solver CFR.
"""

from typing import List, Dict, Optional, Any
import math

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
        bet_to_call: float = 0.0,
        hero_contribution: Optional[float] = None,
        villain_contribution: Optional[float] = None,
        min_raise_increment: Optional[float] = None
    ):
        self.player = player
        self.pot_size = pot_size
        self.hero_stack = hero_stack
        self.villain_stack = villain_stack
        self.board_cards = board_cards
        self.history = history
        self.current_street = current_street
        self.bet_to_call = bet_to_call
        # O pote já contém as contribuições. Stacks são fichas ainda disponíveis.
        # Sem totais observados, usamos um ledger RELATIVO (ator=0, rival=call).
        # Diferenças são conhecidas, mas o total absoluto não pode ser certificado.
        self.contributions_known = hero_contribution is not None and villain_contribution is not None
        if (hero_contribution is None) != (villain_contribution is None):
            raise ValueError("Informe ambas as contribuições da street ou nenhuma.")
        if not self.contributions_known:
            hero_contribution = bet_to_call if player == 1 else 0.0
            villain_contribution = bet_to_call if player == 0 else 0.0
        amounts = (pot_size, hero_stack, villain_stack, bet_to_call, hero_contribution, villain_contribution)
        if any(not math.isfinite(v) or v < 0 for v in amounts):
            raise ValueError("Contexto financeiro da árvore deve ser finito e não negativo.")
        if min_raise_increment is not None and (not math.isfinite(min_raise_increment) or min_raise_increment <= 0):
            raise ValueError("Incremento mínimo conhecido deve ser positivo.")
        self.hero_contribution = hero_contribution
        self.villain_contribution = villain_contribution
        self.current_bet = max(hero_contribution, villain_contribution)
        own = hero_contribution if player == 0 else villain_contribution
        if not math.isclose(max(0, self.current_bet - own), bet_to_call, abs_tol=1e-9):
            raise ValueError("CALL incompatível com as contribuições da street.")
        self.min_raise_increment = min_raise_increment
        self.action_details: Dict[str, Any] = {}
        
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
        own_stack = root.hero_stack if root.player == 0 else root.villain_stack
        if min(root.hero_stack, root.villain_stack) == 0 and (root.bet_to_call == 0 or own_stack == 0):
            self._refund_uncalled(root)
            root.current_street = "SHOWDOWN"
            root.player = 0
            root.bet_to_call = 0
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
        if node.is_terminal() or (min(node.hero_stack, node.villain_stack) == 0 and node.bet_to_call == 0):
            return []
        if node.bet_to_call == 0:
            actions = [ACTION_CALL]  # Atua como CHECK
        else:
            actions = [ACTION_FOLD, ACTION_CALL]
        
        if current_depth < self.max_depth_per_street:
            for action in (ACTION_BET_25, ACTION_BET_50, ACTION_BET_100, ACTION_ALL_IN):
                try:
                    details = self._describe_action(node, action)
                except ValueError:
                    continue
                if details["action_type"] != "CALL":
                    actions.append(action)
                
        return actions

    def _describe_action(self, node: GameNode, action: str) -> Dict[str, Any]:
        """Fonte única do custo avaliado e apresentado.

        BET_q abre q*P sem aposta; diante de aposta, AUMENTA o maior total em
        q*P: investimento adicional = call + q*P. Nunca é um raise-to q*P.
        Mínimo desconhecido permanece explicitamente não verificado.
        """
        stack = node.hero_stack if node.player == 0 else node.villain_stack
        other_stack = node.villain_stack if node.player == 0 else node.hero_stack
        own = node.hero_contribution if node.player == 0 else node.villain_contribution
        details = dict(action_code=action, additional_amount=0.0, raise_to=None,
                       raise_by=0.0, is_all_in=False, short_raise=False,
                       size_verified=True, validation_reason="", uncalled_refund=0.0)
        if node.is_terminal() or stack == 0:
            raise ValueError("Jogador sem fichas ou nó terminal não pode agir.")
        if action == ACTION_FOLD:
            if node.bet_to_call == 0:
                raise ValueError("FOLD indisponível sem aposta a pagar.")
            details["action_type"] = "FOLD"
            return details
        if action == ACTION_CALL or (action == ACTION_ALL_IN and stack <= node.bet_to_call):
            amount = min(stack, node.bet_to_call)
            details.update(action_type="CALL" if amount > 0 else "CHECK", additional_amount=amount,
                           is_all_in=amount == stack)
            return details
        if other_stack == 0:
            raise ValueError("Não há adversário financiável para responder à aposta.")
        if action == ACTION_ALL_IN:
            amount = stack
        elif action in (ACTION_BET_25, ACTION_BET_50, ACTION_BET_100):
            amount = node.bet_to_call + node.pot_size * float(action.split("_")[1]) / 100
            if amount > stack:
                raise ValueError("Raise convencional excede o stack; utilize ALL_IN.")
        else:
            raise ValueError(f"Ação desconhecida: {action}")
        target = own + amount
        increment = target - node.current_bet
        if increment <= 0 or amount <= node.bet_to_call:
            raise ValueError("Aposta deve aumentar efetivamente o total anterior.")
        short = node.min_raise_increment is not None and increment < node.min_raise_increment
        if short and action != ACTION_ALL_IN:
            raise ValueError("Aumento inferior ao mínimo conhecido; apenas all-in curto é permitido.")
        verified = node.min_raise_increment is not None or action == ACTION_ALL_IN
        details.update(action_type="ALL-IN" if action == ACTION_ALL_IN else ("RAISE" if node.bet_to_call else "BET"),
                       additional_amount=amount, raise_to=target if node.contributions_known else None,
                       raise_by=increment, is_all_in=amount == stack,
                       short_raise=short if node.min_raise_increment is not None else None,
                       size_verified=verified,
                       validation_reason="" if verified else "Incremento mínimo da aposta não observado; tamanho não verificado.")
        return details

    @staticmethod
    def _refund_uncalled(node: GameNode) -> float:
        """Heads-up: devolve excesso não coberto, sem construir side pots."""
        excess = abs(node.hero_contribution - node.villain_contribution)
        if excess > node.pot_size:
            raise ValueError("Pote não permite representar a devolução não coberta.")
        node.pot_size -= excess
        if node.hero_contribution > node.villain_contribution:
            node.hero_stack += excess
            node.hero_contribution -= excess
        else:
            node.villain_stack += excess
            node.villain_contribution -= excess
        node.current_bet = max(node.hero_contribution, node.villain_contribution)
        node.bet_to_call = 0
        return excess

    def _apply_action(self, node: GameNode, action: str) -> GameNode:
        details = self._describe_action(node, action)
        next_player = 1 if node.player == 0 else 0
        new_pot = node.pot_size
        new_hero_stack = node.hero_stack
        new_villain_stack = node.villain_stack
        new_bet_to_call = 0.0
        
        contributions = [node.hero_contribution, node.villain_contribution]
        amount = details["additional_amount"]
        new_pot += amount
        contributions[node.player] += amount
        if node.player == 0:
            new_hero_stack -= amount
        else:
            new_villain_stack -= amount
        if action != ACTION_FOLD:
            new_bet_to_call = max(0, contributions[node.player] - contributions[next_player])
        minimum = node.min_raise_increment
        if details["raise_by"] and minimum is not None and not details["short_raise"]:
            minimum = details["raise_by"]

        hist = node.history + "." + action if node.history else action

        child = GameNode(
            player=next_player,
            pot_size=new_pot,
            hero_stack=new_hero_stack,
            villain_stack=new_villain_stack,
            board_cards=node.board_cards,
            history=hist,
            current_street=node.current_street,
            bet_to_call=new_bet_to_call,
            hero_contribution=contributions[0], villain_contribution=contributions[1],
            min_raise_increment=minimum
        )
        child.contributions_known = node.contributions_known
        child.action_details = details
        if details["action_type"] == "CALL" and min(child.hero_stack, child.villain_stack) == 0:
            child.action_details["uncalled_refund"] = self._refund_uncalled(child)
        return child

    def _advance_street(self, node: GameNode) -> GameNode:
        streets = ["PRE_FLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"]
        try:
            idx = streets.index(node.current_street)
            next_street = streets[idx + 1] if idx + 1 < len(streets) else "SHOWDOWN"
        except ValueError:
            next_street = "SHOWDOWN"
            
        if min(node.hero_stack, node.villain_stack) == 0:
            next_street = "SHOWDOWN"  # Runout abstrato já representado pela equity.
        hist = node.history + "|" if node.history else "|"
        
        advanced = GameNode(
            player=0, 
            pot_size=node.pot_size,
            hero_stack=node.hero_stack,
            villain_stack=node.villain_stack,
            board_cards=node.board_cards,
            history=hist,
            current_street=next_street,
            bet_to_call=0.0,
            hero_contribution=node.hero_contribution if next_street == "SHOWDOWN" else 0,
            villain_contribution=node.villain_contribution if next_street == "SHOWDOWN" else 0
        )
        advanced.action_details = dict(node.action_details)
        advanced.contributions_known = node.contributions_known if next_street == "SHOWDOWN" else True
        return advanced
