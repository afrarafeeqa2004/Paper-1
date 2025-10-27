import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = cv2.imread("/content/drive/MyDrive/Colab Notebooks/peppers.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#simulate white balance / calibration errors
blue_cast = img.copy().astype(np.float32)
blue_cast[:, :, 0] *= 1.3  #increase blue channel (cool tone)

yellow_cast = img.copy().astype(np.float32)
yellow_cast[:, :, 2] *= 1.3  #increase red channel (warm tone)

blue_cast = np.clip(blue_cast, 0, 255).astype(np.uint8)
yellow_cast = np.clip(yellow_cast, 0, 255).astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Properly Calibrated (Reference)")
ax[1].imshow(cv2.cvtColor(blue_cast, cv2.COLOR_BGR2RGB))
ax[1].set_title("Improper Calibration – Blue Cast")
ax[2].imshow(cv2.cvtColor(yellow_cast, cv2.COLOR_BGR2RGB))
ax[2].set_title("Improper Calibration – Yellow Cast")
for a in ax: a.axis("off")
plt.suptitle("Example 2: Color Calibration Errors Causing Color Casts")
plt.show()
