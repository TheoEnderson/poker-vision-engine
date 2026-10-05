import cv2
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
cv2.imwrite('debug_hero_spot.png', img[250:350, 750:900])
