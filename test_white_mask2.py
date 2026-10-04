import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))

bt, bl, bw, bh = 415, 734, 449, 109
margin = 15
masked = img_1080.copy()
cv2.rectangle(
    masked,
    (max(0, bl - margin), max(0, bt - margin)),
    (min(1920, bl + bw + margin), min(1080, bt + bh + margin)),
    (0, 0, 0),
    -1,
)

gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
white_u8 = (white_mask.astype(np.uint8)) * 255

cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    cx, cy, cw, ch = cv2.boundingRect(c)
    if cw > 40 and ch > 60:
        print(f"Cand: x={cx}, y={cy}, w={cw}, h={ch}")
