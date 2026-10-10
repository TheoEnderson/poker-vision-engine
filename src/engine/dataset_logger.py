import csv
import json
import datetime
from pathlib import Path
from typing import Dict, Any

class DatasetLogger:
    """
    Registra os cenários matemáticos e estratégicos em um arquivo CSV.
    Mantém histórico completo para Self-Play e Self-Improvement (V6).
    """
    
    def __init__(self, filepath: str = "dataset_partidas.csv"):
        self.filepath = Path(filepath)
        self.headers = [
            "timestamp", 
            "hand_id",
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
            "reasoning",
            "cfr_strategy",
            "financial_result"
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
        hero_str = "-".join(analysis.get("hero_cards", [])) if analysis.get("hero_cards") else ""
        board_str = "-".join(analysis.get("board", [])) if analysis.get("board") else ""
        
        # Garante que temos a estratégia serializada
        cfr_strat = analysis.get("cfr_strategy", {})
        cfr_json = json.dumps(cfr_strat) if cfr_strat else "{}"
        
        # Identificador rudimentar de mão baseado no tempo (pode ser refinado depois)
        hand_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
        
        row = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hand_id,
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
            analysis.get("reason", ""),
            cfr_json,
            "" # Financial result a ser preenchido a posteriori se desejar
        ]

        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
