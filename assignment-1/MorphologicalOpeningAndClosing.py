import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flower.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

titles = ['Original Binary', 'Opening', 'Closing']
images = [img, opening, closing]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.show()
