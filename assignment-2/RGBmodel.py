import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/peppers.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#split the image into its R, G, B channels
b, g, r = cv2.split(img)

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title('Original RGB Image')
plt.axis('off')

#red Channel
plt.subplot(2, 2, 2)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.axis('off')

#green Channel
plt.subplot(2, 2, 3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.axis('off')

#blue Channel
plt.subplot(2, 2, 4)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()
