import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Afra/datasets/archive/hazelnut/test/print/008.png', cv2.IMREAD_GRAYSCALE)

prewitt_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

prewitt_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

grad_x = cv2.filter2D(image, -1, prewitt_x)
grad_y = cv2.filter2D(image, -1, prewitt_y)

magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(np.clip(magnitude, 0, 255))

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(grad_x, cmap='gray')
plt.title('Prewitt X (Vertical edges)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(grad_y, cmap='gray')
plt.title('Prewitt Y (Horizontal edges)')
plt.axis('off')

plt.figure()
plt.imshow(magnitude, cmap='gray')
plt.title('Prewitt Edge Magnitude')
plt.axis('off')

plt.show()
