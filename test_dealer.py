import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
# Convert to HSV to find the red 'D' or white circle
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define range for white (the button background)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])
mask_white = cv2.inRange(hsv, lower_white, upper_white)

# Define range for red (the 'D')
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([180, 255, 255])

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask_red1 + mask_red2

# Find contours in red mask
cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
found = []
for c in cnts:
    area = cv2.contourArea(c)
    if 10 < area < 100: # The 'D' is small
        x, y, w, h = cv2.boundingRect(c)
        # Check if surrounded by white
        if w > 4 and h > 4:
            found.append((x, y, w, h, area))

print("Found red contours:", found)
