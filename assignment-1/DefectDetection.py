import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("pcb.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

edges = cv2.Canny(blur, 50, 150)

kernel = np.ones((3,3), np.uint8)

closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

defect_mask = cv2.subtract(closed, edges)

defect_mask = cv2.dilate(defect_mask, kernel, iterations=1)

contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 30:  
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.putText(output, "Defect", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original PCB (Grayscale)")

plt.subplot(1,3,2)
plt.imshow(defect_mask, cmap='gray')
plt.title("Defect Mask (Edges + Morphology Diff)")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected PCB Defects (Annotated)")
plt.show()
