pip install ultralytics tensorflow keras opencv-python matplotlib

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov5nu.pt")

#load an image
img_path = "/content/product-photo-image-grid-1024x1024.webp"
img = cv2.imread(img_path)

results = model(img)

for r in results:
    annotated = r.plot()  #draw bounding boxes
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

#print detected classes and confidences
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Detected: {model.names[cls]} (Confidence: {conf:.2f})")
