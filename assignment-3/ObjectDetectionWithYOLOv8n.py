pip install ultralytics tensorflow keras opencv-python matplotlib

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

detector = YOLO("yolov8n.pt")
image_file = "E:/Afra/datasets/xxxx.jpg"
frame = cv2.imread(image_file)

output = detector(frame)

for res in output:
    marked = res.plot()
    plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

for obj in output[0].boxes:
    label_id = int(obj.cls[0])
    accuracy = float(obj.conf[0])
    print(f"Object: {detector.names[label_id]} | Confidence: {accuracy:.2f}")
