import cv2
import numpy as np
import matplotlib.pyplot as plt

#load image
thermal_img = cv2.imread('/kaggle/input/image-8/image-6.jpg', cv2.IMREAD_GRAYSCALE)

#normalize for stability
thermal_norm = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX)

#denoise
smooth = cv2.GaussianBlur(thermal_norm, (5, 5), 0)

#adaptive threshold
mean = np.mean(smooth)
std = np.std(smooth)
k = 2

threshold_val = mean + k * std
_, mask = cv2.threshold(smooth, threshold_val, 255, cv2.THRESH_BINARY)

mask = mask.astype(np.uint8)

#clean mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#convert grayscale to color to draw contours
output = cv2.cvtColor(thermal_norm, cv2.COLOR_GRAY2BGR)

#draw contours
cv2.drawContours(output, contours, -1, (255, 255, 0), 2)

cv2.imwrite("thermal_abnormal_zones.png", output)

plt.figure(figsize=(10,6))
plt.imshow(output, cmap='gray')
plt.title("Detected Abnormal Heating Zones")
plt.axis('off')
plt.show()
