import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import math

# Load image
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
h, w = img.shape[:2]
cx, cy = w // 2, h // 2

# 1. Find Dealer Button
template = cv2.imread('assets/templates/dealer_button.png')
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

if max_val > 0.6:
    dealer_x = max_loc[0] + template.shape[1] // 2
    dealer_y = max_loc[1] + template.shape[0] // 2
    dealer_angle = math.degrees(math.atan2(dealer_y - cy, dealer_x - cx))
    print(f"Dealer found at {dealer_x}, {dealer_y} | Angle: {dealer_angle:.1f}°")
else:
    print("Dealer not found!")

# 2. Find Hero
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, otsu = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = pytesseract.image_to_data(otsu, output_type=Output.DICT)
hero_x, hero_y = -1, -1
for i, text in enumerate(data["text"]):
    if "Ment" in text or "End" in text:
        hero_x = data["left"][i] // 2 + data["width"][i] // 4
        hero_y = data["top"][i] // 2 + data["height"][i] // 4
        hero_angle = math.degrees(math.atan2(hero_y - cy, hero_x - cx))
        print(f"Hero found at {hero_x}, {hero_y} | Angle: {hero_angle:.1f}°")
        break

# 3. Calculate position
# Define 6-max ideal angles (Clockwise starting from Mid Right)
# Note: In OpenCV, y goes down, so angles are: 
# Right: 0, Bottom Right: +45, Bottom Left: +135, Left: +180/-180, Top Left: -135, Top Right: -45
ideal_seats_6 = [
    (0, "Mid Right"),
    (45, "Bot Right"),
    (135, "Bot Left"),
    (180, "Mid Left"),
    (-135, "Top Left"),
    (-45, "Top Right")
]

def snap_to_seat(angle, seats):
    # normalize angle to -180 to 180
    angle = (angle + 180) % 360 - 180
    best_seat = None
    min_diff = 999
    for i, (ideal_angle, name) in enumerate(seats):
        diff = abs((angle - ideal_angle + 180) % 360 - 180)
        if diff < min_diff:
            min_diff = diff
            best_seat = i
    return best_seat

dealer_idx = snap_to_seat(dealer_angle, ideal_seats_6)
hero_idx = snap_to_seat(hero_angle, ideal_seats_6)

positions_6max = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
dist = (hero_idx - dealer_idx) % 6
print(f"Distance: {dist} -> Position: {positions_6max[dist]}")
