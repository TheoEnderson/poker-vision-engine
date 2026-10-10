import cv2
import numpy as np

def test_contours(img_path):
    board = cv2.imread(img_path)
    if board is None: return
    
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cards = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 40 < w < 100 and 60 < h < 120:
            cards.append((x, y, w, h))
            
    # Sort by x coordinate
    cards.sort(key=lambda b: b[0])
    print(f"Found {len(cards)} card contours in {img_path}: {cards}")

# Assuming test_board_crop.png is the 449x109 slice
