import cv2
import numpy as np

img = cv2.imread('full_padded.png')
# ReplayPoker board cards have a white background.
# The board is roughly in the center: y=415, x=734.
# Let's crop it and save
board_crop = img[410:410+120, 725:725+465]
cv2.imwrite("test_board_exact.png", board_crop)
