import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"E:/Afra/datasets/archive/hazelnut/test/print/008.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(gray, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(sobelx, cmap='gray'); plt.title("Sobel X"); plt.axis('off')
plt.subplot(2,2,3); plt.imshow(sobely, cmap='gray'); plt.title("Sobel Y"); plt.axis('off')
plt.subplot(2,2,4); plt.imshow(sobel, cmap='gray'); plt.title("Sobel XY"); plt.axis('off')
plt.show()
