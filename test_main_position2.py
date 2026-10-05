import sys
from main import PokerAnalyticsApp
from src.ocr.ocr_reader import extract_hero_position

pos = extract_hero_position('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png', verbose=True)
print("Returned:", pos)
