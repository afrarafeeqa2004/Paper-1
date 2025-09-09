import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Afra Rafeeqa/Pictures/Screenshots/image-4.png', cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian Edge Detection'), plt.axis('off')
plt.show()
