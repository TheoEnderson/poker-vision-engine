"""
test_preflop_and_ocr.py
Ponto de entrada raiz para execução da suíte de testes unitários.
"""

import sys
import unittest
from pathlib import Path

# Adiciona a raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_preflop_and_ocr import TestPreflopAndOCR

if __name__ == '__main__':
    unittest.main()
