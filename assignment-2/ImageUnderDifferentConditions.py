import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

base_folder ="/content/drive/MyDrive/xxxx"
output_folder ="/content/drive/MyDrive/output"
os.makedirs(base_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

print(f"Folders ready:\n- Base: {base_folder}\n- Output: {output_folder}")


def adjust_brightness(img, factor):
    img = img.astype(np.float32)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def directional_light(img, direction='horizontal'):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    if direction == 'horizontal':
        mask = np.tile(np.linspace(0.5, 1.5, w), (h, 1))
    else:
        mask = np.tile(np.linspace(0.5, 1.5, h)[:, np.newaxis], (1, w))
    out = img.astype(np.float32) * mask[:, :, np.newaxis]
    return np.clip(out, 0, 255).astype(np.uint8)

def gaussian_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def adjust_exposure(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

variants = []
for img_name in os.listdir(base_folder):
    img_path = os.path.join(base_folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    variants = [
        ("bright", adjust_brightness(img, 1.5)),
        ("dim", adjust_brightness(img, 0.5)),
        ("directional", directional_light(img, 'horizontal')),
        ("blurred", gaussian_blur(img, 7)),
        ("high_exposure", adjust_exposure(img, 0.5)),
        ("low_exposure", adjust_exposure(img, 2.0))
    ]
    name_base = os.path.splitext(img_name)[0]
    for name, var_img in variants:
        out_path = os.path.join(output_folder, f"{name_base}_{name}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(var_img, cv2.COLOR_RGB2BGR))

print(f"Simulated dataset created in: {output_folder}")
print(f"Total files: {len(os.listdir(output_folder))}")
sample_files = os.listdir(output_folder)[:6]
fig, axes = plt.subplots(1, len(sample_files), figsize=(20, 5))
for i, fname in enumerate(sample_files):
    img = cv2.imread(os.path.join(output_folder, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(fname.split("_")[-1].replace(".jpg", ""))
plt.suptitle("Simulated Industrial Images under Different Conditions", fontsize=16)
plt.show()
