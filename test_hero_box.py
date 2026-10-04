import cv2
import numpy as np
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
x, y, w, h = 749, 230, 75, 57
cv2.imwrite('/dev/shm/hero_cards_raw.png', img[y:y+h, x:x+w])
