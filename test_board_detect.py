import sys
sys.path.append('.')
from src.vision.detector import detect_board
from src.vision.vision import grab_screen_cdp

frame = grab_screen_cdp()
board = detect_board(frame, verbose=True)
print("Detected board:", board)
