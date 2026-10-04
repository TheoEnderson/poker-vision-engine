import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
gray = cv2.cvtColor(img_1080, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img_1080, cv2.COLOR_BGR2HSV)
white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
white_u8 = (white_mask.astype(np.uint8)) * 255
cv2.imwrite('/dev/shm/white_u8.png', white_u8)
