import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/top_left_crop.png')
# Let's search for a small white circle with red inside
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])
mask_white = cv2.inRange(hsv, lower_white, upper_white)

cnts, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    area = cv2.contourArea(c)
    if 300 < area < 800:
        x, y, w, h = cv2.boundingRect(c)
        # Check aspect ratio
        if 0.8 < w/h < 1.2:
            print("Found circle at", x, y, w, h)
            crop = img[y:y+h, x:x+w]
            cv2.imwrite('dealer_template.png', crop)
            break
