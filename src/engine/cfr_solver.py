import random
import math
from typing import Dict, List, Any
from src.engine.game_tree import GameNode, VALID_ACTIONS

class CFRSolver:
    """
    Motor CFR+ (Counterfactual Regret Minimization).
    Calcula estratégias médias para subjogos abstratos.
    Um orçamento finito de iterações não garante explorabilidade zero.
    """
    def __init__(self):
        # Tabela de Arrependimento Cumulativo: info_set -> {action: regret}
        self.cumulative_regrets: Dict[str, Dict[str, float]] = {}
        # Tabela de Estratégia Cumulativa: info_set -> {action: prob_sum}
        self.cumulative_strategy: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        """Normaliza sem arredondar e sem overflow na soma de pesos finitos."""
        if not weights:
            return {}
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError("Pesos da estratégia devem ser finitos e não negativos.")
        scale = max(weights.values())
        if scale == 0:
            return {a: 1.0 / len(weights) for a in weights}
        scaled = {a: w / scale for a, w in weights.items()}
        total = math.fsum(scaled.values())
        return {a: w / total for a, w in scaled.items()}

    def get_strategy(self, info_set: str, valid_actions: List[str]) -> Dict[str, float]:
        """Calcula a Mixed Strategy usando Regret Matching."""
        if info_set not in self.cumulative_regrets:
            self.cumulative_regrets[info_set] = {a: 0.0 for a in valid_actions}
            
        regrets = self.cumulative_regrets[info_set]
        positive_regrets = {a: max(0.0, r) for a, r in regrets.items() if a in valid_actions}
        return self._normalize(positive_regrets)

    def compute_utility(self, node: GameNode, equity: float, initial_hero_stack: float) -> float:
        """
        Retorno líquido do Hero: (equity/100) * pote elegível - investimento.

        Equity é a participação consolidada percentual, inclusive em empates.
        Investimento é a redução do stack disponível desde a raiz, já líquida
        de devoluções da árvore. Contribuições anteriores à raiz não são
        descontadas novamente. Fold equity emerge das respostas na árvore;
        não há bônus de posição, agressividade ou tamanho do pote.
        """
        investment = initial_hero_stack - node.hero_stack
        return (equity / 100.0) * node.pot_size - investment

    def run_cfr(self, node: GameNode, hero_cards: List[str], equity: float, p0: float, p1: float, initial_hero_stack: float) -> float:
        """
        Executa uma iteração de CFR recursivamente na GameTree.
        Retorna a utilidade do nó sob a perspectiva do jogador atual (node.player).
        """
        if node.is_terminal():
            # Se for terminal por FOLD
            if node.history and node.history.endswith("FOLD"):
                # Se next_player é 0, Hero ganha o pote pois Villain foldou.
                if node.player == 0:
                    util_hero = (node.hero_stack + node.pot_size) - initial_hero_stack
                else: # Hero foldou
                    util_hero = node.hero_stack - initial_hero_stack
                return util_hero if node.player == 0 else -util_hero
            
            # Showdown usa a participação consolidada do Hero no pote elegível.
            util_hero = self.compute_utility(node, equity, initial_hero_stack)
            return util_hero if node.player == 0 else -util_hero
            
        info_set = node.get_info_set(hero_cards)
        valid_actions = list(node.children.keys())
        
        if not valid_actions:
            util_hero = self.compute_utility(node, equity, initial_hero_stack)
            return util_hero if node.player == 0 else -util_hero
            
        strategy = self.get_strategy(info_set, valid_actions)
        
        # Atualiza estratégia cumulativa para o jogador atual
        if info_set not in self.cumulative_strategy:
            self.cumulative_strategy[info_set] = {a: 0.0 for a in valid_actions}
            
        prob_weight = p0 if node.player == 0 else p1
        for a in valid_actions:
            self.cumulative_strategy[info_set][a] += strategy[a] * prob_weight
            
        # Calcula utilidade de cada ação
        action_utils = {}
        node_util = 0.0
        
        for a in valid_actions:
            child_node = node.children[a]
            
            if node.player == 0:
                child_util = self.run_cfr(child_node, hero_cards, equity, p0 * strategy[a], p1, initial_hero_stack)
            else:
                child_util = self.run_cfr(child_node, hero_cards, equity, p0, p1 * strategy[a], initial_hero_stack)
                
            util_for_me = child_util if child_node.player == node.player else -child_util
                
            action_utils[a] = util_for_me
            node_util += strategy[a] * util_for_me
            
        # Atualiza os Regrets para o jogador atual
        weight = p1 if node.player == 0 else p0
        for a in valid_actions:
            regret = action_utils[a] - node_util
            new_regret = self.cumulative_regrets[info_set][a] + (weight * regret)
            self.cumulative_regrets[info_set][a] = max(0.0, new_regret) # CFR+
                
        return node_util

    def solve(self, root: GameNode, hero_cards: List[str], equity: float, iterations: int = 1000) -> Dict[str, float]:
        """
        Roda o CFR para o subjogo atual. Retorna a Estratégia Mista (Mixed Strategy) final.
        """
        initial_hero_stack = root.hero_stack
        for _ in range(iterations):
            self.run_cfr(root, hero_cards, equity, 1.0, 1.0, initial_hero_stack)
            
        info_set = root.get_info_set(hero_cards)
        valid_actions = list(root.children.keys())
        
        if not valid_actions:
            return {}
            
        if info_set not in self.cumulative_strategy:
            return {a: 1.0/len(valid_actions) for a in valid_actions}
            
        strategy_sum = self.cumulative_strategy[info_set]
        return self._normalize({a: strategy_sum.get(a, 0.0) for a in valid_actions})
