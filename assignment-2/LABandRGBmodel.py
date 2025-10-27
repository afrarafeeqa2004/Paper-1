import numpy as np
import matplotlib.pyplot as plt
from skimage import color

rgb_1 = np.array([[[70, 130, 180]]], dtype=np.uint8)  
lab_1 = color.rgb2lab(rgb_1 / 255.0)
lab_2 = lab_1.copy()
lab_2[0, 0, 0] += 3     
lab_2[0, 0, 1] += 1
lab_2[0, 0, 2] -= 1     
rgb_2 = (color.lab2rgb(lab_2) * 255).astype(np.uint8)
img_1 = np.ones((100, 100, 3), dtype=np.uint8) * rgb_1
img_2 = np.ones((100, 100, 3), dtype=np.uint8) * rgb_2
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(img_1)
ax[0].set_title(f"RGB Image 1\n{rgb_1[0,0]}")
ax[0].axis("off")

ax[1].imshow(img_2)
ax[1].set_title(f"RGB Image 2\n{rgb_2[0,0]}")
ax[1].axis("off")

plt.show()

print("LAB values:")
print("Image 1 (LAB):", lab_1[0,0])
print("Image 2 (LAB):", lab_2[0,0])
print("\n perceptual difference:", np.linalg.norm(lab_1 - lab_2))
