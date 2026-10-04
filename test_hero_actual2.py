import cv2
from src.vision.detector import find_and_detect_hero_cards
import sys

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
cards = find_and_detect_hero_cards(img_1080, threshold=0.10, verbose=True)
print("HERO CARDS:", cards)
