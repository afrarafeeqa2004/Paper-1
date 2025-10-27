import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/content/drive/MyDrive/Colab Notebooks/peppers.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = image.astype(np.float32) / 255.0

#function to quantize image
def quantize_image(img, n_levels):
    img_q = np.floor(img * (n_levels - 1)) / (n_levels - 1)
    img_q = np.clip(img_q, 0, 1)
    return img_q.astype(np.float32)

#quantize image to different bit levels
levels = [256, 64, 16, 8, 4]
fig, axes = plt.subplots(1, len(levels), figsize=(15, 4))

for i, lvl in enumerate(levels):
    img_q = quantize_image(image, lvl)
    axes[i].imshow(img_q)
    axes[i].axis('off')
    axes[i].set_title(f'{lvl} levels')

plt.suptitle("Color Quantization Artifacts (Posterization Effect)", fontsize=14)
plt.show()
