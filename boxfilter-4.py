import cv2
import matplotlib.pyplot as plt

image = cv2.imread('E:/Afra/datasets/archive/grid/test/broken/001.png', cv2.IMREAD_GRAYSCALE)

box_filtered = cv2.blur(image, (15, 15))

gaussian_filtered = cv2.GaussianBlur(image, (15, 15), 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(box_filtered, cmap='gray')
plt.title("Box Filtered (Simple Averaging)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filtered (Weighted Averaging)")
plt.axis('off')

plt.show()
