import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# The dealer button is very bright white
_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
cv2.imwrite('debug_white.png', thresh)

# Let's find contours in the white threshold
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
circles = []
for c in cnts:
    area = cv2.contourArea(c)
    # the dealer button should have an area around pi*r^2. If r=15, area=700. If r=10, area=314.
    if 200 < area < 800:
        (x, y), radius = cv2.minEnclosingCircle(c)
        # Check if it's reasonably circular
        circle_area = np.pi * (radius**2)
        if 0.7 < area / circle_area < 1.3:
            # Let's check if there is red inside
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
            mean_color = cv2.mean(img, mask=mask)
            # mean_color is (B, G, R, alpha)
            # Red should be higher than Blue and Green inside the circle because of the 'D'
            if mean_color[2] > 200 and mean_color[2] > mean_color[0] + 20 and mean_color[2] > mean_color[1] + 20:
                circles.append((int(x), int(y), int(radius), mean_color))
            
print("Found dealer circles:", circles)
