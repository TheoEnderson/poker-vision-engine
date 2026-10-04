import csv
import os
import datetime
from pathlib import Path
from typing import Dict, Any

class DatasetLogger:
    """
    Registra os cenários matemáticos e de visão computacional em um arquivo CSV.
    Esse dataset será usado futuramente para treinar modelos de Machine Learning (XGBoost/Redes Neurais)
    para prever a lucratividade de ações específicas (EV Real vs EV Teórico).
    """
    
    def __init__(self, filepath: str = "dataset_partidas.csv"):
        self.filepath = Path(filepath)
        self.headers = [
            "timestamp", 
            "state", 
            "hero_cards", 
            "board_cards", 
            "pot_size", 
            "bet_to_call", 
            "hero_stack", 
            "pot_odds_pct", 
            "equity_pct", 
            "ev_chips", 
            "recommendation",
            "reasoning"
        ]
        self._initialize_file()

    def _initialize_file(self):
        """Cria o arquivo CSV e os cabeçalhos caso não exista."""
        if not self.filepath.exists():
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_state(self, analysis: Dict[str, Any]):
        """
        Registra um estado recalculado no CSV.
        """
        # Formata as cartas em uma string limpa
        hero_str = "-".join(analysis.get("hero_cards", [])) if analysis.get("hero_cards") else ""
        board_str = "-".join(analysis.get("board", [])) if analysis.get("board") else ""
        
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis.get("state", ""),
            hero_str,
            board_str,
            analysis.get("pot_size", 0.0),
            analysis.get("bet_to_call", 0.0),
            analysis.get("hero_stack", 0.0),
            analysis.get("pot_odds", 0.0),
            analysis.get("equity", 0.0),
            analysis.get("ev", 0.0),
            analysis.get("action", ""),
            analysis.get("reason", "")
        ]

        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
