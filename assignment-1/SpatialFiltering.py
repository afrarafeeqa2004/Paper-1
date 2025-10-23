import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("E:/Afra/datasets/archive/tile/test/crack/012.png", 0)

blurred = cv2.GaussianBlur(img, (5, 5), 0)

#apply high-pass filter
laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

#enhance by combining with original image
enhanced = cv2.addWeighted(img, 1, laplacian, 1, 0)

#threshold to isolate defects
defects = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

cv2.imwrite("enhanced_surface.png", enhanced)
cv2.imwrite("highlighted_defects.png", defects)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(enhanced, cmap='gray'); plt.title("Enhanced Surface"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(defects, cmap='gray'); plt.title("Highlighted Defects"); plt.axis('off')
plt.show()
