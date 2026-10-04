import cv2
import numpy as np
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
hx, hy, hw, hh = 1403, 430, 142, 118
hero_roi = img_1080[hy:hy+hh, hx:hx+hw]
cw = int(hw / 2)
c1_roi = hero_roi[:, :cw]
c1_corner = c1_roi[0:min(hh, max(60, int(hh*0.55))), 0:int(c1_roi.shape[1]*0.35)]

cv2.imwrite('/dev/shm/c1_corner.png', c1_corner)
print("Saved c1_corner.png")
