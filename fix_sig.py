import re

with open("src/engine/risk_engine.py", "r") as f:
    content = f.read()

old = """def calculate_decision_score(
    equity: float, 
    pot_odds: float, 
    ev: float, 
    spr: float, 
    texture: Dict[str, bool],
    tier: int,
    bet_ratio: float,
    has_pair_or_better: bool
) -> float:"""

new = """def calculate_decision_score(
    equity: float, 
    pot_odds: float, 
    ev: float, 
    spr: float, 
    texture: Dict[str, bool],
    tier: int,
    bet_ratio: float,
    has_pair_or_better: bool,
    state: str = "PRE_FLOP"
) -> float:"""

content = content.replace(old, new)

with open("src/engine/risk_engine.py", "w") as f:
    f.write(content)
