import random
from typing import Dict, List

class PreflopBlueprint:
    """
    Camada 1: Blueprint Strategy para Pré-flop.
    Abordagem O(1) usando tabelas heurísticas pré-computadas baseadas em Tiers de Força,
    em vez de computar CFR ao vivo no Pré-Flop (o que explodiria a árvore).
    """
    
    # Mapeia (Tier, Ação_Anterior_Oponente) -> {Ações: Probabilidade}
    # Tiers (1=Premium, 2=Strong, 3=Playable, 4=Trash)
    # bet_to_call == 0 (Unopened) vs bet_to_call > 0 (Facing a bet)
    
    STRATEGY_UNOPENED = {
        1: {"BET_100": 0.8, "BET_50": 0.2},      # Sempre sobe agressivo com Premium
        2: {"BET_100": 0.4, "BET_50": 0.6},      # Sobe forte com Strong
        3: {"BET_50": 0.3, "CALL": 0.4, "FOLD": 0.3}, # Misto com Playable (Limp/Mini-raise/Fold)
        4: {"FOLD": 1.0}                         # Fold Trash
    }
    
    STRATEGY_FACING_BET = {
        1: {"ALL_IN": 0.3, "BET_100": 0.6, "CALL": 0.1}, # 3-bet / All-in
        2: {"BET_100": 0.2, "CALL": 0.7, "FOLD": 0.1},   # Majoritariamente flat call, raramente 3-bet
        3: {"CALL": 0.3, "FOLD": 0.7},                   # Paga raramente se estiver com pot odds, mas maioria fold
        4: {"FOLD": 1.0}                                 # Fold Trash
    }

    @classmethod
    def get_strategy(cls, tier: int, bet_to_call: float, spr: float) -> Dict[str, float]:
        """Retorna a Mixed Strategy pré-flop."""
        
        # Se stack estiver muito baixo (SPR muito curto) e tier bom, vamos de All-in
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
