pip install ultralytics tensorflow keras opencv-python matplotlib

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import requests

image_url = "E:/Afra/datasets/xxxxx.jpg"
response = requests.get(image_url)
with open("bus.jpg", "wb") as f:
    f.write(response.content)

det = YOLO("yolov8n.pt")
img = cv2.imread("bus.jpg")
res1 = det(img)
out1 = res1[0].plot()
plt.imshow(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB))
plt.title("Object Detection")
plt.axis("off")
plt.show()

seg = YOLO("yolov8n-seg.pt")
res2 = seg(img)
out2 = res2[0].plot()
plt.imshow(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB))
plt.title("Image Segmentation")
plt.axis("off")
plt.show()

