import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:/Afra/datasets/archive/cable/test/missing_wire/002.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(img, (15,15), 10)

hpf_kernel = np.array([[-1, -1, -1],
                       [-1,  8.5, -1],
                       [-1, -1, -1]])

hpf_img = cv2.filter2D(blurred, -1, hpf_kernel)

enhanced = cv2.add(blurred, hpf_img)

plt.figure(figsize=(12,6))
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title('Original'), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(blurred, cmap='gray'), plt.title('Blurred Image'), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(enhanced, cmap='gray'), plt.title('Enhanced with HPF'), plt.axis('off')
plt.show()
