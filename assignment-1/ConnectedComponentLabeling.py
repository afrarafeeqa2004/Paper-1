import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("pcb.jpg", cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img, (5, 5), 0)

_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if np.sum(thresh == 255) > np.sum(thresh == 0):
    thresh = cv2.bitwise_not(thresh)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

good_count = 0
defective_count = 0

areas = stats[1:, cv2.CC_STAT_AREA]
normal_area = np.median(areas)


for i in range(1, n_labels): 
    x, y, w, h, area = stats[i]

    comp_mask = (labels == i).astype("uint8") * 255
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        continue

    perimeter = cv2.arcLength(contours[0], True)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

    if 0.7*normal_area <= area <= 1.5*normal_area and circularity > 0.6:
        label = "Good"
        color = (0, 255, 0)  
        good_count += 1
    else:
        label = "Defective"
        color = (0, 0, 255)   
        defective_count += 1

    
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

total_joints = good_count + defective_count
good_pct = 100.0 * good_count / total_joints if total_joints > 0 else 0
bad_pct = 100.0 * defective_count / total_joints if total_joints > 0 else 0

print(f"Total solder joints: {total_joints}")
print(f"Good joints: {good_count} ({good_pct:.1f}%)")
print(f"Defective joints: {defective_count} ({bad_pct:.1f}%)")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original PCB")

plt.subplot(1,2,2)
plt.imshow(output[:,:,::-1])
plt.title("Solder Joint Inspection")
plt.show()
