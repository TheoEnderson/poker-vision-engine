import sys
from src.vision.detector import detect_board
from src.config import SAMPLES_DIR

import os
for f in os.listdir(SAMPLES_DIR):
    if f.endswith('.png'):
        print(f"\n--- Testing {f} ---")
        detect_board(str(SAMPLES_DIR / f), verbose=True)
