import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('fabric.jpg', 0) 

_, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV) 

kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

titles = ['Original', 'Binary Mask', 'After Opening', 'After Closing']
images = [img, binary, opening, closing]

plt.figure(figsize=(16,4))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
