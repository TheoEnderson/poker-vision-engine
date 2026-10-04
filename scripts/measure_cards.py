import cv2
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791148647049_7a42dccf.png')

h, w = img.shape[:2]
cx1, cx2 = int(w*0.3), int(w*0.7)
cy1, cy2 = int(h*0.3), int(h*0.7)
center = img[cy1:cy2, cx1:cx2]

gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imwrite('/dev/shm/thresh.png', thresh)

cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    x, y, cw, ch = cv2.boundingRect(c)
    if cw > 20 and ch > 30:
        print(f"Card cand: x={x+cx1}, y={y+cy1}, w={cw}, h={ch}")
