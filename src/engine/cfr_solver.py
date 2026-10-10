import random
from typing import Dict, List, Any
from src.engine.game_tree import GameNode, VALID_ACTIONS

class CFRSolver:
    """
    Motor CFR+ (Counterfactual Regret Minimization).
    Gera Mixed Strategies inexploráveis para subjogos.
    """
    def __init__(self):
        # Tabela de Arrependimento Cumulativo: info_set -> {action: regret}
        self.cumulative_regrets: Dict[str, Dict[str, float]] = {}
        # Tabela de Estratégia Cumulativa: info_set -> {action: prob_sum}
        self.cumulative_strategy: Dict[str, Dict[str, float]] = {}

    def get_strategy(self, info_set: str, valid_actions: List[str]) -> Dict[str, float]:
        """Calcula a Mixed Strategy usando Regret Matching."""
        if info_set not in self.cumulative_regrets:
            self.cumulative_regrets[info_set] = {a: 0.0 for a in valid_actions}
            
        regrets = self.cumulative_regrets[info_set]
        positive_regrets = {a: max(0.0, r) for a, r in regrets.items() if a in valid_actions}
        sum_positive = sum(positive_regrets.values())
        
        if sum_positive > 0:
            return {a: r / sum_positive for a, r in positive_regrets.items()}
        else:
            n = len(valid_actions)
            return {a: 1.0 / n for a in valid_actions}

    def compute_utility(self, node: GameNode, equity: float, initial_hero_stack: float) -> float:
        """
        Calcula a Função de Utilidade Completa no nó terminal a partir da perspectiva do Hero.
        Utility = Expected_Final_Chips - Initial_Chips + Heuristics
        """
        p_win = equity / 100.0
        p_lose = 1.0 - p_win
        
        # Expected Final Chips do Hero
        expected_final_chips = (p_win * (node.hero_stack + node.pot_size)) + (p_lose * node.hero_stack)
        
        # Lucro líquido esperado (Chip EV real)
        chip_ev = expected_final_chips - initial_hero_stack
        
        # Fold Equity (Heurística baseada no histórico de agressão)
        fold_equity = 0.0
        if not node.is_terminal() or "FOLD" not in node.history:
            if "ALL_IN" in node.history:
                fold_equity = node.pot_size * 0.40
            elif "BET_100" in node.history:
                fold_equity = node.pot_size * 0.30
            elif "BET_50" in node.history:
                fold_equity = node.pot_size * 0.15
            elif "BET_25" in node.history:
                fold_equity = node.pot_size * 0.05
                
        # Positional Value (Hero geralmente atua depois = vantagem)
        positional_value = node.pot_size * 0.05
        
        utility = chip_ev + fold_equity + positional_value
        return utility

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
            
            # Se for showdown, o utilitário do Hero é dado pela heurística.
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
        total = sum(strategy_sum.values())
        
        if total > 0:
            return {a: round(v / total, 3) for a, v in strategy_sum.items()}
            
        return {a: round(1.0/len(valid_actions), 3) for a in valid_actions}
