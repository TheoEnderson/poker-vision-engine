"""
src/config.py
Configuração centralizada do Poker Analytics Engine (PAE).

Centraliza caminhos de diretórios, ROIs (Region of Interest) da tela,
limiares de visão computacional, parâmetros de OCR e regras do motor de risco.
"""

import os
from pathlib import Path
from typing import Dict

# Caminhos Fundamentais do Projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
TEMPLATES_DIR = ASSETS_DIR / "templates"
RANKS_DIR = TEMPLATES_DIR / "ranks"
SUITS_DIR = TEMPLATES_DIR / "suits"
SAMPLES_DIR = ASSETS_DIR / "samples"
CARDS_DIR = ASSETS_DIR / "cards"
UNKNOWN_CARDS_DIR = ASSETS_DIR / "unknown_cards"

# Identificador do Hero na Mesa
HERO_IDENTIFIER: str = os.getenv("HERO_IDENTIFIER", "The_Ment_End")

# Regiões de Interesse (ROIs) na Tela (Resolução padrão da mesa)
POT_ROI: Dict[str, int] = {
    "top": int(os.getenv("POT_ROI_TOP", 210)),
    "left": int(os.getenv("POT_ROI_LEFT", 850)),
    "width": int(os.getenv("POT_ROI_WIDTH", 220)),
    "height": int(os.getenv("POT_ROI_HEIGHT", 60)),
}

ACTION_BUTTONS_ROI: Dict[str, int] = {
    "top": int(os.getenv("ACTION_ROI_TOP", 870)),
    "left": int(os.getenv("ACTION_ROI_LEFT", 1200)),
    "width": int(os.getenv("ACTION_ROI_WIDTH", 650)),
    "height": int(os.getenv("ACTION_ROI_HEIGHT", 150)),
}

# Limiares de Visão Computacional
DEFAULT_RANK_THRESHOLD: float = float(os.getenv("RANK_THRESHOLD", 0.68))
DEFAULT_SUIT_THRESHOLD: float = float(os.getenv("SUIT_THRESHOLD", 0.68))
CARD_BRIGHTNESS_THRESHOLD: int = int(os.getenv("CARD_BRIGHTNESS_THRESHOLD", 150))
MAX_GREEN_RATIO: float = float(os.getenv("MAX_GREEN_RATIO", 0.40))

# Limiares e Sanidade de OCR
MAX_REALISTIC_POT: float = float(os.getenv("MAX_REALISTIC_POT", 10000.0))
OCR_WHITELIST_DIGITS: str = "0123456789.,"

# Regras de Risco e Gestão de Stack
HEAVY_BET_STACK_RATIO: float = float(os.getenv("HEAVY_BET_STACK_RATIO", 0.40))
MICRO_BET_STACK_RATIO: float = float(os.getenv("MICRO_BET_STACK_RATIO", 0.02))
PREFLOP_SPECULATIVE_BET_RATIO: float = float(os.getenv("PREFLOP_SPEC_RATIO", 0.025))

# Parâmetros Padrão do Jogo e Monte Carlo
DEFAULT_HERO_STACK: float = float(os.getenv("DEFAULT_HERO_STACK", 1000.0))
DEFAULT_NUM_OPPONENTS: int = int(os.getenv("DEFAULT_NUM_OPPONENTS", 4))
MONTE_CARLO_ITERATIONS: int = int(os.getenv("MONTE_CARLO_ITERATIONS", 10000))
POLL_INTERVAL: float = float(os.getenv("POLL_INTERVAL", 0.5))
