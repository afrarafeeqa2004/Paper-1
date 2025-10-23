import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("E:/Afra/datasets/archive/bottle/test/broken_large/004.png", 0)

#otsu's thresholding
otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#adaptive thresholding
adaptive_mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imwrite("otsu_mask.png", otsu_mask)
cv2.imwrite("adaptive_mask.png", adaptive_mask)

plt.figure(figsize=(10,6))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(otsu_mask, cmap='gray'); plt.title("otsu_mask"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(adaptive_mask, cmap='gray'); plt.title("adaptive_mask"); plt.axis('off')
plt.show()

