import cv2
import numpy as np
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
hx, hy = 1403, 430
c1_corner = img_1080[hy:hy+64, hx:hx+24]

hsv = cv2.cvtColor(c1_corner, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(c1_corner, cv2.COLOR_BGR2GRAY)
green_mask = cv2.inRange(hsv, np.array([30, 50, 30]), np.array([90, 255, 255]))

print("Green mask sum:", np.sum(green_mask))

target_pixels = (green_mask == 0) & (gray < 185)
print("Target pixels sum:", np.sum(target_pixels))
