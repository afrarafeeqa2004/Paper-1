import cv2
import numpy as np
import os
from skimage.exposure import match_histograms


site_a_ref_path = 'angle-1.png'
site_b_images = ['light-1.png', 'light-2.png', 'light-3.png', 'angle-2.png', 'angle-3.png']

def preprocess_for_classifier(img_path, ref_img):
    if not os.path.exists(img_path):
        print(f"Skipping: {img_path} not found.")
        return None

    img = cv2.imread(img_path)

    #resize to match reference dimensions
    img = cv2.resize(img, (ref_img.shape[1], ref_img.shape[0]))
    #lighting stabilization
    img_stable_light = match_histograms(img, ref_img, channel_axis=-1)

    #feature enhancement
    gray = cv2.cvtColor(img_stable_light.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return enhanced

reference_img = cv2.imread(site_a_ref_path)

if reference_img is None:
    print(f"Error: Could not load reference image {site_a_ref_path}. Check the filename.")
else:
    processed_dataset = []
    for img_path in site_b_images:
        processed = preprocess_for_classifier(img_path, reference_img)
        if processed is not None:
            processed_dataset.append(processed)
            cv2.imwrite(f'stabilized_{img_path}', processed)

    print(f"Successfully processed {len(processed_dataset)} images.")
