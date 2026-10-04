import cv2
import numpy as np
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
hx, hy, hw, hh = 1403, 430, 142, 118
c1_corner = img_1080[hy:hy+64, hx:hx+24]
gray = cv2.cvtColor(c1_corner, cv2.COLOR_BGR2GRAY)
print("Max gray:", np.max(gray))
print("Mean gray:", np.mean(gray))
