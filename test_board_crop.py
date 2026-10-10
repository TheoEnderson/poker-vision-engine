import cv2
import numpy as np
from src.vision.vision import grab_screen_cdp

frame = grab_screen_cdp()
cv2.imwrite("full_padded.png", frame)
board_crop = frame[415:415+109, 734:734+449]
cv2.imwrite("test_board_crop.png", board_crop)
