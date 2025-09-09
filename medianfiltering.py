import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Afra/datasets/image-1.jpg', cv2.IMREAD_GRAYSCALE)

def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy()
    rows, cols = img.shape
    
    num_salt = int(prob * rows * cols / 2)
    y_coords = np.random.randint(0, rows, num_salt)
    x_coords = np.random.randint(0, cols, num_salt)
    noisy[y_coords, x_coords] = 255

    num_pepper = int(prob * rows * cols / 2)
    y_coords = np.random.randint(0, rows, num_pepper)
    x_coords = np.random.randint(0, cols, num_pepper)
    noisy[y_coords, x_coords] = 0

    return noisy

noisy_image = add_salt_pepper_noise(image, prob=0.1)

median_filtered = cv2.medianBlur(noisy_image, 3)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(median_filtered, cmap="gray")
plt.title("After Median Filter")
plt.axis("off")

plt.show()
