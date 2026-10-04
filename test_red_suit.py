import cv2
import numpy as np

img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791149690115_38df9f54.png')
img_1080 = cv2.resize(img, (1920, 1080))
h_card = img_1080[430:430+118, 1403:1403+118]
gray = cv2.cvtColor(h_card, cv2.COLOR_BGR2GRAY)
print("Min gray:", np.min(gray))
print("Mean gray:", np.mean(gray))

# Save the gray card
cv2.imwrite('/dev/shm/h_card_gray.png', gray)
