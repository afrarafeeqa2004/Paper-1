!git clone https://github.com/cuilimeng/CrackForest-dataset.git

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_shape=(256, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b1 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b1 = layers.Conv2D(256, 3, activation='relu', padding='same')(b1)

    # Decoder
    u1 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(b1)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)

    u2 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(64, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)
    return model

unet = unet_model()
unet.summary()

import os
image_dir = "/content/CrackForest-dataset/image"
mask_dir = "/content/CrackForest-dataset/mask"

image_dir = "/content/CrackForest-dataset/image"
mask_dir = "/content/CrackForest-dataset/groundTruth"

import os
print("Images:", len(os.listdir(image_dir)))
print("Masks:", len(os.listdir(mask_dir)))

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

image_dir = "/content/CrackForest-dataset/image"
mask_dir = "/content/CrackForest-dataset/groundTruth"

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

dataset = CrackDataset(image_dir, mask_dir, transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

IMG_SIZE = (256,256)

def load_image(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

def create_dataset(image_paths, mask_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

common = sorted(list(image_names & mask_names))

image_files = [os.path.join(IMAGE_DIR, f"{name}.jpg") for name in common]
mask_files = [os.path.join(MASK_DIR, f"{name}.jpg") for name in common]

split = int(0.8 * len(image_files))
train_img, val_img = image_files[:split], image_files[split:]
train_mask, val_mask = mask_files[:split], mask_files[split:]

import tensorflow as tf

# Convert all image & mask paths to strings explicitly
train_img = [str(p) for p in train_img]
train_mask = [str(p) for p in train_mask]
val_img = [str(p) for p in val_img]
val_mask = [str(p) for p in val_mask]

def load_image(image_path, mask_path):
    image_path = tf.cast(image_path, tf.string)
    mask_path = tf.cast(mask_path, tf.string)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256))
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask


#build TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_mask))
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(8).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_mask))
val_ds = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(8).prefetch(tf.data.AUTOTUNE)


import os
import numpy as np
import cv2
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

image_dir = "/content/CrackForest-dataset/image"
mask_dir = "/content/CrackForest-dataset/groundTruth"

IMG_SIZE = (256, 256)

image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])

mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.mat')])

images = []
masks = []

for img_path, mask_path in zip(image_files, mask_files):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    # Load mask from .mat file
    mat = loadmat(mask_path)
    mask = mat['groundTruth'][0][0][1] if 'groundTruth' in mat else list(mat.values())[-1]
    mask = cv2.resize(mask, IMG_SIZE)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask / np.max(mask)

    images.append(img)
    masks.append(mask)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)

print("Loaded images:", images.shape)
print("Loaded masks:", masks.shape)

#split dataset
train_img, val_img, train_mask, val_mask = train_test_split(images, masks, test_size=0.2, random_state=42)

#TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_mask)).batch(8).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_mask)).batch(8).prefetch(tf.data.AUTOTUNE)

unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = unet.fit(train_ds, validation_data=val_ds, epochs=50)


unet.save("/content/unet_crack_segmentation.h5")

unet.save("/content/unet_crack_segmentation.keras")


unet.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning UNet50...")
unet.fit(train_ds, validation_data=val_ds, epochs=2)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#load your trained model
unet = load_model("/content/unet_crack_segmentation.keras")

#load and preprocess one test image
test_path = "/content/CrackForest-dataset/image/055.jpg"
img = cv2.imread(test_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize to same size as training
img_resized = cv2.resize(img_rgb, (256, 256))
img_norm = img_resized / 255.0

pred = unet.predict(np.expand_dims(img_norm, axis=0))
mask_pred = (pred[0] > 0.22).astype(np.uint8)

kernel = np.ones((2, 2), np.uint8)

mask_clean = cv2.morphologyEx(mask_pred, cv2.MORPH_OPEN, kernel, iterations=1)

mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Raw Prediction")
plt.imshow(mask_pred.squeeze(), cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("After Improved Cleaning")
plt.imshow(mask_clean.squeeze(), cmap='gray')
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Predicted Crack Mask")
plt.imshow(mask_pred.squeeze(), cmap='gray')
plt.axis("off")
plt.show()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Raw Prediction")
plt.imshow(mask_pred.squeeze(), cmap='gray')
plt.axis("off")

plt.subplot(1,2,2)
plt.title("After Improved Cleaning")
plt.imshow(mask_clean.squeeze(), cmap='gray')
plt.axis("off")
plt.show()
