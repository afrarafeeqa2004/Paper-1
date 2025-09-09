import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("E:/Afra/datasets/archive/toothbrush/test/defective/024.png", cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)  
laplacian = cv2.convertScaleAbs(laplacian) 

alpha = 1
enhanced = cv2.addWeighted(image, 1, laplacian, alpha, 0)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Document")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Edges")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(enhanced, cmap='gray')
plt.title("Edge Enhanced Document")
plt.axis('off')

plt.show()
