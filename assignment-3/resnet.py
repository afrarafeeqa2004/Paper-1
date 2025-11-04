from google.colab import files
files.upload()

!pip install kaggle pandas scikit-learn

!kaggle datasets download -d yidazhang07/bridge-cracks-image -p ./data

import zipfile, os

for file in os.listdir('./data'):
    if file.endswith('.zip'):
        with zipfile.ZipFile('./data/' + file, 'r') as zip_ref:
            zip_ref.extractall('./data')

print("Dataset extracted successfully")
import os
for root, dirs, files in os.walk('./data'):
    print(root, "->", len(files), "files")
from sklearn.model_selection import train_test_split
import glob
import shutil

#create train/test directories
os.makedirs('./data/train', exist_ok=True)
os.makedirs('./data/test', exist_ok=True)

import glob, shutil
from sklearn.model_selection import train_test_split

#base directory (dataset root)
base_dir = './data'

#split directories
train_dir = './data/train_split'
val_dir = './data/val_split'
test_dir = './data/test_split'

for folder in [train_dir, val_dir, test_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

image_paths = []
for ext in ['jpg', 'jpeg', 'png']:
    image_paths.extend(glob.glob(os.path.join(base_dir, '**', f'*.{ext}'), recursive=True))

image_paths = [p for p in image_paths if 'train_split' not in p and 'val_split' not in p and 'test_split' not in p]

print(f"Total images found: {len(image_paths)}")

if len(image_paths) == 0:
    raise ValueError("No images found.")

x_train, x_temp = train_test_split(image_paths, test_size=0.3, random_state=42)
x_val, x_test = train_test_split(x_temp, test_size=0.5, random_state=42)

y_train = [None] * len(x_train)
y_val = [None] * len(x_val)
y_test = [None] * len(x_test)

def safe_copy(file_list, dest_dir):
    for path in file_list:
        filename = os.path.basename(path)
        dest = os.path.join(dest_dir, filename)
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copy(path, dest)

safe_copy(x_train, train_dir)
safe_copy(x_val, val_dir)
safe_copy(x_test, test_dir)

print(f"x_train: {len(x_train)} images")
print(f"x_val:   {len(x_val)} images")
print(f"x_test:  {len(x_test)} images")
print("Dataset split successfully and consistent across runs!")

import os, shutil, glob

base_dir = "/content/data/train_split"

crack_dir = os.path.join(base_dir, "Cracked")
noncrack_dir = os.path.join(base_dir, "NonCracked")
os.makedirs(crack_dir, exist_ok=True)
os.makedirs(noncrack_dir, exist_ok=True)

for file in glob.glob(os.path.join(base_dir, "*.jpg")):
    fname = os.path.basename(file)
    if "_test" in fname:
        shutil.move(file, os.path.join(crack_dir, fname))
    elif "_temp" in fname:
        shutil.move(file, os.path.join(noncrack_dir, fname))

print("Images sorted into 'Cracked' and 'NonCracked' folders successfully")

for split in ["val_split", "test_split"]:
    base_dir = f"/content/data/{split}"
    crack_dir = os.path.join(base_dir, "Cracked")
    noncrack_dir = os.path.join(base_dir, "NonCracked")
    os.makedirs(crack_dir, exist_ok=True)
    os.makedirs(noncrack_dir, exist_ok=True)

    for file in glob.glob(os.path.join(base_dir, "*.jpg")):
        fname = os.path.basename(file)
        if "_test" in fname:
            shutil.move(file, os.path.join(crack_dir, fname))
        elif "_temp" in fname:
            shutil.move(file, os.path.join(noncrack_dir, fname))

    print(f"Sorted {split} folder.")

import tensorflow as tf
from tensorflow.keras import layers, models

#paths and constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = "/content/data/train_split"
val_dir = "/content/data/val_split"
test_dir = "/content/data/test_split"

#load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

#save class names before normalization
class_names = train_ds.class_names
print("Detected classes:", class_names)

#normalize and optimize datasets
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

#ResNet50 Transfer Learning
base_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_resnet.trainable = False  # freeze backbone

model = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")  # number of classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#train and fine-tune
print("Training ResNet50...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

print("Fine-tuning last few layers...")
base_resnet.trainable = True
for layer in base_resnet.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=2)

import numpy as np
from tensorflow.keras.preprocessing import image

#example: load one image from your test folder
img_path = "/content/crackimage.jpg"  # update path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

pred = model.predict(img_array)

#convert to readable class name
predicted_class = class_names[np.argmax(pred)]
print("Predicted class:", predicted_class)
