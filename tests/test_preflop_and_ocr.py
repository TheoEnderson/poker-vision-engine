"""
tests/test_preflop_and_ocr.py
Suíte de testes de sanidade para leitura de OCR (teto de pote) e regras de sobrevivência de stack (Risk Engine).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

# Adiciona a raiz do projeto ao sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.ocr.ocr_reader import extract_pot
    from src.engine.risk_engine import make_decision
except ImportError:
    from ocr_reader import extract_pot
    from risk_engine import make_decision


class TestPreflopAndOCR(unittest.TestCase):

    @patch('pytesseract.image_to_string')
    def test_ocr_sanity_limit(self, mock_tesseract):
        """Garante que leituras astronômicas (> MAX_REALISTIC_POT) sejam rejeitadas pelo filtro de sanidade."""
        mock_tesseract.return_value = "93700.0"
        dummy_img = np.zeros((800, 1200, 3), dtype=np.uint8)

        pot_value = extract_pot(image_path_or_frame=dummy_img, default_value=123.0, verbose=False)
        self.assertEqual(pot_value, 123.0, "O teto de sanidade deveria ter bloqueado o valor 93700.0")

    def test_heavy_bet_preflop_trash(self):
        """Garante FOLD imediato com mão lixo pré-flop sob aposta pesada (>40% do stack)."""
        decision = make_decision(
            p_win=10.0, p_lose=90.0, p_tie=0.0,
            pot_size=150.0,
            bet_to_call=1235.0,
            hero_stack=1335.0,
            state="PRE_FLOP",
            hero_cards=["2h", "4h"],
            board_cards=[],
            outs=0
        )
        self.assertEqual(decision["action"], "FOLD", "Deveria forçar FOLD com lixo pré-flop sob aposta pesada.")

    def test_heavy_bet_postflop_high_card(self):
        """Garante FOLD imediato com carta alta pós-flop sob aposta que compromete o stack."""
        decision = make_decision(
            p_win=25.0, p_lose=75.0, p_tie=0.0,
            pot_size=150.0,
            bet_to_call=1000.0,
            hero_stack=2000.0,
            state="FLOP",
            hero_cards=["Qh", "6h"],
            board_cards=["4c", "3c", "5c"],
            outs=0
        )
        self.assertEqual(decision["action"], "FOLD", "Deveria forçar FOLD com Carta Alta no pós-flop sob aposta pesada.")


if __name__ == '__main__':
    unittest.main()
