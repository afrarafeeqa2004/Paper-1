import cv2
import numpy as np

def classify_crack(mask,
                   min_pixels=200,        # increased to prevent texture false positives
                   min_component_size=150, # remove noisy components
                   narrow_th=2.0,
                   medium_th=6.0):

    #clean noise-remove tiny components
    mask_uint8 = (mask * 255).astype(np.uint8)

    #connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)

    #keep only components above size threshold
    clean = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_component_size:
            clean[labels == i] = 255

    #count crack pixels after cleaning
    crack_pixels = np.sum(clean > 0)

    #detect no-crack
    if crack_pixels < min_pixels:
        return "No Crack", 0.0

    #compute width using distance transform
    dist = cv2.distanceTransform((clean > 0).astype(np.uint8), cv2.DIST_L2, 5)
    width = dist.max() * 2.0

    #classify severity
    if width < narrow_th:
        severity = "Narrow"
    elif width < medium_th:
        severity = "Medium"
    else:
        severity = "Wide"

    return severity, width


#test
img = cv2.imread("/content/crackimage-48.jpeg", 0)

edges = cv2.Canny(img, 80, 180)
_, mask = cv2.threshold(edges, 10, 1, cv2.THRESH_BINARY)

severity, width = classify_crack(mask)

print("Classification:", severity)
print("Width:", width)
