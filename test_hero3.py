import cv2
import pytesseract
from pytesseract import Output
img = cv2.imread('/home/theo-enderson/.gemini/antigravity/brain/928dc12e-2fa5-44a6-9b6a-890bc8762e01/current_screenshot.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
_, otsu = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
data = pytesseract.image_to_data(otsu, output_type=Output.DICT)
for i, text in enumerate(data["text"]):
    if "Ment" in text or "End" in text:
        left = data["left"][i] // 2
        top = data["top"][i] // 2
        print(f"Text '{text}' found at {left}, {top}")
