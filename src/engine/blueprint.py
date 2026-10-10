import json
import random
from pathlib import Path
from typing import Dict

class PreflopBlueprint:
    """
    Camada 1: Blueprint Strategy para Pré-flop.
    Pode carregar uma estratégia treinada offline via JSON (Self-Improvement).
    """
    
    STRATEGY_UNOPENED = {
        1: {"BET_100": 0.8, "BET_50": 0.2},
        2: {"BET_100": 0.4, "BET_50": 0.6},
        3: {"BET_50": 0.3, "CALL": 0.4, "FOLD": 0.3},
        4: {"FOLD": 1.0}
    }
    
    STRATEGY_FACING_BET = {
        1: {"ALL_IN": 0.3, "BET_100": 0.6, "CALL": 0.1},
        2: {"BET_100": 0.2, "CALL": 0.7, "FOLD": 0.1},
        3: {"CALL": 0.3, "FOLD": 0.7},
        4: {"FOLD": 1.0}
    }
    
    _loaded = False

    @classmethod
    def load_trained_blueprint(cls, filepath: str = "blueprint_trained.json"):
        if cls._loaded:
            return
            
        path = Path(filepath)
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                if "STRATEGY_UNOPENED" in data:
                    cls.STRATEGY_UNOPENED = {int(k): v for k, v in data["STRATEGY_UNOPENED"].items()}
                if "STRATEGY_FACING_BET" in data:
                    cls.STRATEGY_FACING_BET = {int(k): v for k, v in data["STRATEGY_FACING_BET"].items()}
                cls._loaded = True
            except Exception as e:
                print(f"Erro ao carregar blueprint treinado: {e}")

    @classmethod
    def get_strategy(cls, tier: int, bet_to_call: float, spr: float) -> Dict[str, float]:
        cls.load_trained_blueprint()
        
        if spr < 2.0 and tier <= 2:
            return {"ALL_IN": 1.0}
            
        if bet_to_call == 0:
            return cls.STRATEGY_UNOPENED.get(tier, {"FOLD": 1.0})
        else:
            return cls.STRATEGY_FACING_BET.get(tier, {"FOLD": 1.0})
            
    @classmethod
    def sample_action(cls, strategy: Dict[str, float]) -> str:
        actions = list(strategy.keys())
        probs = list(strategy.values())
        return random.choices(actions, weights=probs, k=1)[0]
