import cv2
import numpy as np
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
hx, hy, hw, hh = 1403, 430, 142, 118
hero_roi = img_1080[hy:hy+hh, hx:hx+hw]
cw = int(hw / 2)
c1_roi = hero_roi[:, :cw]
c1_corner = c1_roi[0:min(hh, max(60, int(hh*0.55))), 0:int(c1_roi.shape[1]*0.35)]

# Find the minimum gray value inside c1_corner
gray = cv2.cvtColor(c1_corner, cv2.COLOR_BGR2GRAY)
print("Min gray in corner 1:", np.min(gray))
print("Mean gray in corner 1:", np.mean(gray))

# Let's see the colors of the minimum gray pixel
min_loc = np.unravel_index(np.argmin(gray), gray.shape)
print("Color at min gray:", c1_corner[min_loc])
