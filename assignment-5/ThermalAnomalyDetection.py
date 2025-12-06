import cv2
import numpy as np

def detect_abnormal_hotspots(img_path, save_path="abnormal_hotspots.png", 
                             k=2.0, min_area=50):#k = number of standard deviations above mean to classify abnormal heat

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")

    img_blur = cv2.GaussianBlur(img, (7,7), 0)

    #threshold
    mean = np.mean(img_blur)
    std  = np.std(img_blur)
    thresh_value = mean + k * std

    _, thresh = cv2.threshold(img_blur, thresh_value, 255, cv2.THRESH_BINARY)

    #clean small regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    #find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #abnormal hotspots
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    abnormal_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # skip small normal heat
        
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x,y), (x+w, y+h), (0,255,255), 2)

        #draw contour
        cv2.drawContours(overlay, [cnt], -1, (0,255,255), 1)

        abnormal_count += 1

    cv2.imwrite(save_path, overlay)

    return abnormal_count, overlay

count, _ = detect_abnormal_hotspots(
    "/kaggle/input/image-9/image-6.jpg",
    "abnormal_hotspots.png",
    k=2.5,        # stricter threshold (reduces humans / normal heat)
    min_area=120  # removes tiny false positives
)

print("Abnormal zones detected:", count)
