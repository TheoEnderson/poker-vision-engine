import unittest
from src.engine.risk_engine import make_decision

class TestRiskEngineCFR(unittest.TestCase):
    def test_preflop_blueprint(self):
        result = make_decision(
            p_win=80.0, p_lose=20.0, p_tie=0.0,
            pot_size=10.0, bet_to_call=0.0, hero_stack=1000.0,
            state="PRE_FLOP", hero_cards=["Ad", "As"], board_cards=[]
        )
        self.assertIn("Blueprint Strategy", result["reason"])

    def test_postflop_cfr(self):
        result = make_decision(
            p_win=80.0, p_lose=20.0, p_tie=0.0,
            pot_size=50.0, bet_to_call=0.0, hero_stack=1000.0,
            state="FLOP", hero_cards=["Ad", "As"], board_cards=["2h", "7d", "9c"]
        )
        self.assertIn("CFR+ Subgame Solver", result["reason"])

if __name__ == '__main__':
    unittest.main()
