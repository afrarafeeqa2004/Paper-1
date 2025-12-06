import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img = cv2.imread("/content/image-12.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#vertical line enhancement
kernel = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
], dtype=np.float32)

line_response = cv2.filter2D(gray, -1, kernel)

#normalize for visibility
line_norm = cv2.normalize(line_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#threshold to isolate crack
_, thresh = cv2.threshold(line_norm, 200, 255, cv2.THRESH_BINARY)

#morphology to connect
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

#keep only thin, long contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    
    #keep only long thin structures (cracks)
    if w < 20 and h > 80:
        cv2.drawContours(output, [c], -1, (0, 0, 255), 2)

cv2_imshow(line_norm)
cv2_imshow(thresh)
