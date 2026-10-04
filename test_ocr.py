import cv2
from src.ocr.ocr_reader import detect_turn_and_bet_to_call
import sys

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
res = detect_turn_and_bet_to_call(img_1080, verbose=True)
print("OCR:", res)
