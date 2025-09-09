import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Afra Rafeeqa/Pictures/Screenshots/image-4.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Check the image path.")
ksize = 5
gaussian_sigma = 1.2
box = cv2.blur(img, (ksize, ksize))
gauss = cv2.GaussianBlur(img, (ksize, ksize), gaussian_sigma)
def psnr(a, b):
    return cv2.PSNR(a, b)

psnr_box = psnr(img, box)
psnr_gauss = psnr(img, gauss)
plt.figure(figsize=(12, 6))
plt.subplot(1,3,1); plt.imshow(img, cmap='gray');   plt.title('Original'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(box, cmap='gray');   plt.title(f'Box Blur (k={ksize}×{ksize})\nPSNR={psnr_box:.2f} dB'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(gauss, cmap='gray'); plt.title(f'Gaussian (k={ksize}×{ksize}, σ={gaussian_sigma})\nPSNR={psnr_gauss:.2f} dB'); plt.axis('off')
plt.tight_layout(); plt.show()
diff_box   = cv2.absdiff(img, box)
diff_gauss = cv2.absdiff(img, gauss)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(diff_box,   cmap='gray'); plt.title('|Original - Box|');   plt.axis('off')
plt.subplot(1,2,2); plt.imshow(diff_gauss, cmap='gray'); plt.title('|Original - Gaussian|'); plt.axis('off')
plt.tight_layout(); plt.show()
