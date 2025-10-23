import cv2
import numpy as np
import matplotlib.pyplot as plt

#HSV ranges for fruits
fruit_hsv_ranges = {
    "banana": {
        "ripe":   ([20, 100, 100], [30, 255, 255]),   # yellow
        "unripe": ([35, 50, 50],   [85, 255, 255])    # green
    },
    "tomato": {
        "ripe":   ([0, 100, 100],  [10, 255, 255]),   # red
        "unripe": ([35, 50, 50],   [85, 255, 255])    # green
    },
    "mango": {
        "ripe":   ([15, 100, 100], [30, 255, 255]),   # yellow-orange
        "unripe": ([35, 50, 50],   [85, 255, 255])    # green
    },
    "apple": {
        "ripe":   ([0, 100, 100],  [10, 255, 255]),   # red
        "unripe": ([35, 50, 50],   [85, 255, 255])    # green
    },
    "guava": {
        "ripe":   ([20, 50, 80],   [35, 255, 255]),   # light yellow-green
        "unripe": ([35, 80, 80],   [85, 255, 255])    # dark green
    }
}

def classify_fruit(image_path, fruit_type, show=True):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_ripe, upper_ripe = map(np.array, fruit_hsv_ranges[fruit_type]["ripe"])
    lower_unripe, upper_unripe = map(np.array, fruit_hsv_ranges[fruit_type]["unripe"])

    mask_ripe = cv2.inRange(hsv, lower_ripe, upper_ripe)
    mask_unripe = cv2.inRange(hsv, lower_unripe, upper_unripe)

    ripe_pixels = cv2.countNonZero(mask_ripe)
    unripe_pixels = cv2.countNonZero(mask_unripe)

    status = "RIPE" if ripe_pixels > unripe_pixels else "UNRIPE"
    print(f"{fruit_type.capitalize()} → {status}")

    if show:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original ({fruit_type})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_ripe, cmap="gray")
        plt.title("Ripe Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(mask_unripe, cmap="gray")
        plt.title("Unripe Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

samples = {
    "banana": "E:/Afra/datasets/banana-4.jpg",
    "tomato": "E:/Afra/datasets/tomato.jfif",
    "mango": "E:/Afra/datasets/mango-4.jpg",
    "apple": "E:/Afra/datasets/apple.jpg",
    "guava": "E:/Afra/datasets/guava-4.jpg"
}

for fruit, path in samples.items():
    classify_fruit(path, fruit, show=True)
