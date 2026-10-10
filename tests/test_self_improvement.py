import unittest
import os
from src.engine.dataset_logger import DatasetLogger
from src.engine.train_blueprint import run_self_improvement
from src.engine.blueprint import PreflopBlueprint

class TestSelfImprovement(unittest.TestCase):
    def test_pipeline(self):
        logger = DatasetLogger("test_dataset.csv")
        logger.log_state({
            "state": "PRE_FLOP",
            "hero_cards": ["Ah", "As"],
            "pot_size": 10.0,
            "bet_to_call": 0.0,
            "ev": 5.0,
            "action": "BET_100",
            "recommendation": "BET_100",
            "cfr_strategy": {"BET_100": 1.0}
        })
        
        run_self_improvement("test_dataset.csv", "test_blueprint_trained.json")
        self.assertTrue(os.path.exists("test_blueprint_trained.json"))
        
        # Carrega pra ver se não quebra
        PreflopBlueprint.load_trained_blueprint("test_blueprint_trained.json")
        strat = PreflopBlueprint.get_strategy(1, 0.0, 10.0)
        self.assertTrue(len(strat) > 0)
        
        if os.path.exists("test_dataset.csv"):
            os.remove("test_dataset.csv")
        if os.path.exists("test_blueprint_trained.json"):
            os.remove("test_blueprint_trained.json")

if __name__ == '__main__':
    unittest.main()
