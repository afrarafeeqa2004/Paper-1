import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Afra/datasets/archive/hazelnut/test/print/008.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (9,9), 10)

mask = cv2.subtract(image, blurred)

k = 10
unsharp = cv2.addWeighted(image, 1, mask, k, 0)


high_boost = cv2.addWeighted(image, A, mask, 1, 0)

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(unsharp, cmap='gray')
plt.title("Unsharp Masking")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(high_boost, cmap='gray')
plt.title("High-Boost Filtering")
plt.axis('off')

plt.show()
