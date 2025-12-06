!pip install opencv-python scikit-image matplotlib numpy

import cv2
import numpy as np

#load image
img = cv2.imread("/kaggle/input/crack-image/crackimage.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#pre-processing
blur = cv2.GaussianBlur(gray, (5,5), 0)

#adaptive threshold
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21, 5
)

#remove noise, smooth defects
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
clean = cv2.dilate(clean, kernel, iterations=1)

#connected components (finding defects)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean)

#areas of components:
areas = stats[:, cv2.CC_STAT_AREA]

#overlay defects on original image
output = img.copy()

for i in range(1, num_labels):
    area = areas[i]

    if area < 20:    # ignore tiny noise
        continue

    x, y, w, h, _ = stats[i]

    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(output, f"Defect {area}",
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)

cv2.imwrite("binary_mask.png", clean)
cv2.imwrite("detected_defects.png", output)

print("Done.")
