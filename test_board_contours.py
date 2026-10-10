import cv2
import numpy as np
from pathlib import Path
from src.config import SAMPLES_DIR

def test_board(filename):
    print(f"Testing {filename}")
    path = SAMPLES_DIR / filename
    if not path.exists(): return
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    
    # Board region
    board_roi = img[350:650, 450:1450]
    gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(board_roi, cv2.COLOR_BGR2HSV)
    white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
    white_u8 = (white_mask.astype(np.uint8)) * 255
    
    cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        cbx, cby, cbw, cbh = cv2.boundingRect(c)
        if 2000 < area < 15000 and 30 < cbw < 150 and 40 < cbh < 150:
            candidates.append((cbx, cby, cbw, cbh, area))
    
    candidates.sort(key=lambda x: x[0])
    for cand in candidates:
        print(f"  Card: {cand}")

import os
for f in os.listdir(SAMPLES_DIR):
    if f.endswith('.png'):
        test_board(f)

