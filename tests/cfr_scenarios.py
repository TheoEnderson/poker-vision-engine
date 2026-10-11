"""Cenários determinísticos compartilhados por testes e benchmark 4E-B3.

O jogo matricial é um harness abstrato: apenas seu payoff e information sets
são substituídos. Regret matching, recursão, atualização e média são reais.
"""

import math

from src.engine.cfr_solver import CFRSolver
from src.engine.game_tree import GameNode, GameTreeBuilder


ITERATIONS = (10, 30, 50, 100, 250, 500, 1000)
HERO = ["Jh", "9d"]
MATRIX = ((1., -1.), (-2., 1.))


def poker_root(street="RIVER", call=20, depth=1):
    board = ["As", "Kd", "Qc", "7h", "2s"]
    board = board[:{"FLOP": 3, "TURN": 4, "RIVER": 5}[street]]
    root = GameNode(0, 100, 500, 480, board, current_street=street, bet_to_call=call)
    return GameTreeBuilder(depth).build_tree(root)


class MatrixNode:
    def __init__(self, player, info_set, utility=None):
        self.player = player
        self.info_set = info_set
        self.utility = utility
        self.children = {}
        self.hero_stack = 0
        self.history = ""

    def get_info_set(self, hero_cards):
        return self.info_set

    def is_terminal(self):
        return self.utility is not None


class MatrixSolver(CFRSolver):
    def compute_utility(self, node, equity, initial_hero_stack):
        return node.utility


def matrix_root():
    root = MatrixNode(0, "ROW")
    for i, row in enumerate(MATRIX):
        opponent = MatrixNode(1, "COLUMN")  # Não observa a escolha da linha.
        opponent.children = {f"C{j}": MatrixNode(0, "TERMINAL", payoff)
                             for j, payoff in enumerate(row)}
        root.children[f"R{i}"] = opponent
    return root


def fixed_returns_root(returns):
    """Escolha contra retornos esperados fixos, sem aprender fold frequencies."""
    root = MatrixNode(0, "FIXED_RETURNS")
    root.children = {action: MatrixNode(0, "TERMINAL", value)
                     for action, value in returns.items()}
    return root


def average_policy(solver, node):
    actions = list(node.children)
    weights = solver.cumulative_strategy.get(node.get_info_set(HERO), {})
    total = math.fsum(weights.get(a, 0.) for a in actions)
    return {a: weights.get(a, 0.) / total if total else 1 / len(actions) for a in actions}


def hero_terminal_value(solver, node, equity, initial):
    value = solver.run_cfr(node, HERO, equity, 1., 1., initial)
    return value if node.player == 0 else -value


def full_information_gap(root, solver, equity):
    """NashConv exato da árvore abstrata com histórias totalmente observáveis.

    Melhor resposta de Hero contra a média rival menos valor da média Hero
    contra melhor resposta rival. Não aplicar ao harness matricial oculto.
    """
    def best_response(node, responder):
        if node.is_terminal() or not node.children:
            return hero_terminal_value(solver, node, equity, root.hero_stack)
        values = {a: best_response(child, responder) for a, child in node.children.items()}
        if node.player == responder:
            return (max if responder == 0 else min)(values.values())
        strategy = average_policy(solver, node)
        return math.fsum(strategy[a] * values[a] for a in values)
    return best_response(root, 0) - best_response(root, 1)


def matrix_metrics(solver, root):
    row = average_policy(solver, root)
    column = average_policy(solver, root.children["R0"])
    p, q = row["R0"], column["C0"]
    best_row = max(q * r[0] + (1 - q) * r[1] for r in MATRIX)
    best_column = min(p * MATRIX[0][j] + (1 - p) * MATRIX[1][j] for j in range(2))
    return dict(row_p=p, column_q=q, row_l1=2 * abs(p - .6),
                column_l1=2 * abs(q - .4), nash_conv=best_row - best_column)


def count_nodes(root):
    return 1 + sum(count_nodes(child) for child in root.children.values())


def scenarios():
    negative = poker_root(depth=0)
    positive = poker_root(depth=0)
    always_calls = poker_root(call=0)
    always_folds = poker_root(call=0)
    for root, response in ((always_calls, "CALL"), (always_folds, "FOLD")):
        check = root.children["CALL"]
        check.children = {"CALL": check.children["CALL"]}
        bet = root.children["BET_25"]
        bet.children = {response: bet.children[response]}
        root.children = {"CALL": check, "BET_25": bet}
    cases = {
        "call_negative": (negative, CFRSolver, 15),
        "call_positive": (positive, CFRSolver, 25),
        "bet_always_called": (always_calls, CFRSolver, 0),
        "bet_always_folded": (always_folds, CFRSolver, 0),
        "matrix_mixed": (matrix_root(), MatrixSolver, 0),
        "river": (poker_root(), CFRSolver, 40),
        "turn": (poker_root("TURN"), CFRSolver, 40),
        "flop": (poker_root("FLOP"), CFRSolver, 40),
    }
    cases["known_returns"] = (fixed_returns_root({"A": 3., "B": 7.}), MatrixSolver, 0)
    # Payoffs terminais reais de BET_25, equity zero: fold=100, call=-25.
    # A mistura exógena é integrada analiticamente apenas neste harness.
    for name, frequency in (("bluff_insufficient", .1), ("bluff_break_even", .2),
                            ("bluff_sufficient", .4)):
        value = frequency * 100 - (1-frequency) * 25
        cases[name] = (fixed_returns_root({"CHECK": 0., "BET": value}), MatrixSolver, 0)
    return cases
