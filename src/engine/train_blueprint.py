import json
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.engine.blueprint import PreflopBlueprint
from src.engine.preflop_tier import get_preflop_tier

def run_self_improvement(dataset_path: str = "dataset_partidas.csv", output_path: str = "blueprint_trained.json"):
    """
    Simula o Pipeline de Self-Improvement (V6).
    Consome o dataset de mãos passadas e ajusta as probabilidades do Blueprint (Pré-flop).
    Na arquitetura real, rodaria MCCFR em um cluster. Aqui fazemos um update empírico.
    """
    if not os.path.exists(dataset_path):
        print("Nenhum dataset encontrado para treinar.")
        return

    # Estruturas para acumular ações sugeridas e EVs em cada situação
    # tier -> bet_facing -> action -> count
    observations = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["state"] == "PRE_FLOP":
                hero_cards = row["hero_cards"].split("-") if row["hero_cards"] else []
                if not hero_cards:
                    continue
                    
                tier = get_preflop_tier(hero_cards)
                bet_to_call = float(row["bet_to_call"])
                facing_bet = "YES" if bet_to_call > 0 else "NO"
                
                # Para simplificar o aprendizado empírico, se a ação tomada foi acompanhada de EV > 0
                # reforçamos o peso dessa ação.
                ev = float(row["ev_chips"])
                action = row["recommendation"]
                
                # Mapeia de volta pro abstract action pra simplificar (aproximação)
                abstract_action = action
                if action == "BET" or action == "RAISE":
                    abstract_action = "BET_50" # Simplificação do fallback
                elif action == "CHECK":
                    abstract_action = "CALL"
                    
                # Se a mão teve EV positivo ou foi neutra (fold), aumenta o peso empírico
                if ev >= 0 or abstract_action == "FOLD":
                    observations[facing_bet][tier][abstract_action] += 1

    # Copia as estratégias base
    new_unopened = {k: v.copy() for k, v in PreflopBlueprint.STRATEGY_UNOPENED.items()}
    new_facing_bet = {k: v.copy() for k, v in PreflopBlueprint.STRATEGY_FACING_BET.items()}
    
    # Atualiza as estratégias com o aprendizado (blending 80% original / 20% empírico)
    learning_rate = 0.2

    for facing_bet in ["NO", "YES"]:
        target_dict = new_facing_bet if facing_bet == "YES" else new_unopened
        for tier in range(1, 5):
            obs = observations[facing_bet][tier]
            total_obs = sum(obs.values())
            
            if total_obs > 0:
                # Transforma contagem em probabilidade
                emp_probs = {a: count / total_obs for a, count in obs.items()}
                
                # Blending
                for a in target_dict[tier].keys():
                    base_p = target_dict[tier][a]
                    emp_p = emp_probs.get(a, 0.0)
                    target_dict[tier][a] = round((base_p * (1 - learning_rate)) + (emp_p * learning_rate), 3)
                    
                # Adiciona ações que não estavam na base mas foram observadas
                for a in emp_probs.keys():
                    if a not in target_dict[tier]:
                        target_dict[tier][a] = round(emp_probs[a] * learning_rate, 3)
                        
                # Normaliza
                total_prob = sum(target_dict[tier].values())
                if total_prob > 0:
                    for a in target_dict[tier].keys():
                        target_dict[tier][a] = round(target_dict[tier][a] / total_prob, 3)
                        
    # Salva no JSON
    out_data = {
        "STRATEGY_UNOPENED": new_unopened,
        "STRATEGY_FACING_BET": new_facing_bet
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)
        
    print(f"Self-Improvement concluído! Blueprint atualizado salvo em: {output_path}")

if __name__ == "__main__":
    run_self_improvement()
