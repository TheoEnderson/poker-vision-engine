import cv2
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
# Show coordinates around rasul1349 (Top Left)
# It seems the Dealer button is near the avatar.
# Let's crop a 200x200 box around (x=450, y=250)
crop = img[220:300, 360:440]
cv2.imwrite('debug_dealer_crop.png', crop)
