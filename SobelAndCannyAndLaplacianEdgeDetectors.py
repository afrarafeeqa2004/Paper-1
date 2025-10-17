import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("E:/Afra/datasets/archive/hazelnut/test/cut/013.png", 0)

#sobel edge detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)   
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)   
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

#laplacian edge detection
laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=5)
laplacian = cv2.convertScaleAbs(laplacian)

#canny edge detection
canny = cv2.Canny(img, 100, 200)

plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()
