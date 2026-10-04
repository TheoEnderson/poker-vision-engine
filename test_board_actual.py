import cv2
from src.vision.detector import detect_board
import sys

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791148647049_7a42dccf.png')
# Scale it back up to 1920x1080 since the chat UI downscaled it to 1024x576
img_1080 = cv2.resize(img, (1920, 1080))
cards = detect_board(img_1080, verbose=True)
print("CARDS:", cards)
