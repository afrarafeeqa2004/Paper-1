import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Afra/datasets/archive/tile/test/crack/010.png', cv2.IMREAD_GRAYSCALE)

kernel_size = 5  

max_filtered = cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

min_filtered = cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(max_filtered, cmap='gray')
plt.title("Max Filter (Bright Spot Detection)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(min_filtered, cmap='gray')
plt.title("Min Filter (Dark Spot Detection)")
plt.axis('off')

plt.show()
