import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

image = cv2.imread("/content/drive/MyDrive/Colab Notebooks/peppers.png")

lab_image = color.rgb2lab(image)

if image.shape[-1] == 3:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
lab_image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

l_channel = lab_image[:,:,0]
a_channel = lab_image[:,:,1]
b_channel = lab_image[:,:,2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#display the L* channel
axes[0].imshow(l_channel, cmap='gray')
axes[0].set_title('L* Channel (Lightness)')
axes[0].axis('off')

#display the a* channel
axes[1].imshow(a_channel, cmap='gray')
axes[1].set_title('a* Channel (Green-Red)')
axes[1].axis('off')

#display the b* channel
axes[2].imshow(b_channel, cmap='gray')
axes[2].set_title('b* Channel (Blue-Yellow)')
axes[2].axis('off')

plt.tight_layout()
plt.show()
