"""
src/config.py
Configuração centralizada do Poker Analytics Engine (PAE).

Centraliza caminhos de diretórios, ROIs (Region of Interest) da tela,
limiares de visão computacional, parâmetros de OCR e regras do motor de risco.
"""

import os
import json
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
SITES_CONFIG_PATH = PROJECT_ROOT / "sites.json"

# Carregamento Dinâmico de Perfis de Site (Platform Profiles)
ACTIVE_SITE = os.getenv("POKER_SITE", "ReplayPoker")
site_config = {}

if SITES_CONFIG_PATH.exists():
    try:
        with open(SITES_CONFIG_PATH, "r") as f:
            all_sites = json.load(f)
            site_config = all_sites.get(ACTIVE_SITE, {})
    except Exception as e:
        print(f"[AVISO] Erro ao carregar sites.json: {e}")

# Helpers para acessar config com fallback
def get_roi_config(roi_name: str, default_values: dict) -> dict:
    site_roi = site_config.get(roi_name, {})
    return {
        "top": int(os.getenv(f"{roi_name}_TOP", site_roi.get("top", default_values["top"]))),
        "left": int(os.getenv(f"{roi_name}_LEFT", site_roi.get("left", default_values["left"]))),
        "width": int(os.getenv(f"{roi_name}_WIDTH", site_roi.get("width", default_values["width"]))),
        "height": int(os.getenv(f"{roi_name}_HEIGHT", site_roi.get("height", default_values["height"]))),
    }

def get_threshold(name: str, default_val: float, is_int: bool = False):
    site_thresh = site_config.get("THRESHOLDS", {}).get(name, default_val)
    val = os.getenv(name, site_thresh)
    return int(val) if is_int else float(val)

# Identificador do Hero na Mesa
HERO_IDENTIFIER: str = os.getenv("HERO_IDENTIFIER", "The_Ment_End")

# Compatibilidade de SO / Captura de Tela
FORCE_X11_MSS: bool = os.getenv("FORCE_X11_MSS", "False").lower() in ["true", "1", "yes"]

# Regiões de Interesse (ROIs) na Tela (Resolução padrão da mesa)
POT_ROI: Dict[str, int] = get_roi_config("POT_ROI", {"top": 210, "left": 850, "width": 220, "height": 60})

ACTION_BUTTONS_ROI: Dict[str, int] = get_roi_config("ACTION_BUTTONS_ROI", {"top": 870, "left": 1200, "width": 650, "height": 150})

# Limiares de Visão Computacional
DEFAULT_RANK_THRESHOLD: float = get_threshold("RANK_THRESHOLD", 0.68)
DEFAULT_SUIT_THRESHOLD: float = get_threshold("SUIT_THRESHOLD", 0.68)
CARD_BRIGHTNESS_THRESHOLD: int = get_threshold("CARD_BRIGHTNESS_THRESHOLD", 150, True)
MAX_GREEN_RATIO: float = get_threshold("MAX_GREEN_RATIO", 0.40)

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
