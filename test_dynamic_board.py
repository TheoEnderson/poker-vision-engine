import cv2
import numpy as np

def find_board_dynamically(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # Crop central area
    cx1, cx2 = int(w*0.3), int(w*0.7)
    cy1, cy2 = int(h*0.3), int(h*0.7)
    center = img[cy1:cy2, cx1:cx2]
    
    # Binarize to find white cards
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cards = []
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 40 and ch > 60 and ch > cw: # Typical card ratio
            cards.append((x + cx1, y + cy1, cw, ch))
            
    if not cards:
        print("No cards found")
        return None
        
    cards.sort(key=lambda b: b[0])
    
    # Get bounding box of all cards
    min_x = cards[0][0]
    min_y = min([b[1] for b in cards])
    max_x = cards[-1][0] + cards[-1][2]
    max_y = max([b[1] + b[3] for b in cards])
    
    print(f"Dynamic ROI: bt={min_y}, bl={min_x}, bw={max_x-min_x}, bh={max_y-min_y}")
    
find_board_dynamically('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/.user_uploaded/media_1791148647049_7a42dccf.png')
