import cv2
import numpy as np
import base64
import requests
import json

def get_screenshot():
    response = requests.get('http://127.0.0.1:9222/json')
    tabs = response.json()
    page = next((t for t in tabs if t.get('type') == 'page' and not t['url'].startswith('devtools://')), None)
    if not page: return None
    
    ws_url = page['webSocketDebuggerUrl']
    from websocket import create_connection
    ws = create_connection(ws_url)
    ws.send(json.dumps({"id": 1, "method": "Page.captureScreenshot", "params": {"format": "png"}}))
    res = json.loads(ws.recv())
    ws.close()
    
    img_data = base64.b64decode(res['result']['data'])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

img = get_screenshot()
if img is not None:
    cv2.imwrite('test_screen.png', img)
    print(f"Screenshot saved: {img.shape}")
    
    # Let's try to detect the board dynamically
    h, w = img.shape[:2]
    # Restrict to board area roughly
    board_roi = img[350:650, 450:1450]
    cv2.imwrite('board_roi.png', board_roi)
    
    gray = cv2.cvtColor(board_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(board_roi, cv2.COLOR_BGR2HSV)
    white_mask = (gray > 175) & (hsv[:, :, 1] < 55)
    white_u8 = (white_mask.astype(np.uint8)) * 255
    cv2.imwrite('board_white_mask.png', white_u8)
    
    cnts, _ = cv2.findContours(white_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        cbx, cby, cbw, cbh = cv2.boundingRect(c)
        # Board cards are about 60-90 px wide, 80-120 px tall, area 3000-10000
        if 2000 < area < 12000 and 30 < cbw < 120 and 40 < cbh < 150:
            candidates.append((cbx, cby, cbw, cbh))
            
    print(f"Found {len(candidates)} board card candidates")
    for b in sorted(candidates, key=lambda x: x[0]):
        print(b)
else:
    print("Failed to get screenshot")
