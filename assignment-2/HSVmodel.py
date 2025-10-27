import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/peppers.png")

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#split the HSV image into H, S, V channels
h, s, v = cv2.split(img_hsv)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')

#hue Channel
plt.subplot(2, 2, 2)
plt.imshow(h, cmap='hsv')
plt.title('Hue Channel')
plt.axis('off')

#saturation Channel
plt.subplot(2, 2, 3)
plt.imshow(s, cmap='gray')
plt.title('Saturation Channel')
plt.axis('off')

#value Channel
plt.subplot(2, 2, 4)
plt.imshow(v, cmap='gray')
plt.title('Value Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
