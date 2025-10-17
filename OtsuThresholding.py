import cv2
from matplotlib import pyplot as plt

img = cv2.imread("E:/Afra/datasets/archive/bottle/test/broken_large/014.png", 0)

#global thresholding
global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#otsu thresholding
otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Metal Surface")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(global_thresh, cmap='gray')
plt.title("Global Thresholding (127)")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(adaptive_thresh, cmap='gray')
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Thresholding")
plt.axis("off")

plt.tight_layout()
plt.show()
