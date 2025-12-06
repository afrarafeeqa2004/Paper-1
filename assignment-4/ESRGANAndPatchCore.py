import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras import layers, models as keras_models

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#ESRGAN model
def dense_block(x, filters, growth=32):
    concat_feat = [x]
    for _ in range(5):
        out = layers.Conv2D(growth, 3, padding='same', activation='relu')(x)
        concat_feat.append(out)
        x = layers.Concatenate()(concat_feat)
    out = layers.Conv2D(filters, 3, padding='same')(x)
    return layers.Add()([out, concat_feat[0]])

def rrdb_block(x, filters):
    out = dense_block(x, filters)
    out = dense_block(out, filters)
    out = dense_block(out, filters)
    return layers.Add()([x, out * 0.2])

def build_esrgan_generator(num_rrdb=3, filters=64, input_shape=(None, None, 3)):
    inputs = layers.Input(shape=input_shape)
    fea = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = fea
    for _ in range(num_rrdb):
        x = rrdb_block(x, filters)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.Add()([fea, x])
    for _ in range(2):
        x = layers.Conv2D(filters*4, 3, padding='same')(x)
        x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
    return keras_models.Model(inputs, outputs)

generator = build_esrgan_generator()

#patchcore backbone
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

class SimpleDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted(glob.glob(folder + "/*"))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return img, self.files[idx]  # return path to identify later

#folders
train_folder = "/content/drive/MyDrive/xxxx/train"
test_folder  = "/content/drive/MyDrive/xxxx/test"

train_dataset = SimpleDataset(train_folder)
test_dataset  = SimpleDataset(test_folder)

#patchcore transform
tf_patchcore = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

#ESRGAN processing
def esrgan_process(img_pil):
    img = np.array(img_pil).astype(np.float32)
    img = img / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    sr = generator.predict(img)[0]
    sr = (sr+1.0)*127.5
    sr = np.clip(sr,0,255).astype(np.uint8)
    return Image.fromarray(sr)

#extract features
def extract_features(dataset):
    feats = []
    paths = []
    for img, path in dataset:
        sr_img = esrgan_process(img)
        sr_img = tf_patchcore(sr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(sr_img).cpu().numpy()
        feat = feat.reshape(feat.shape[0], -1)  # flatten
        feats.append(feat)
        paths.append(path)
    feats = np.concatenate(feats, axis=0)
    return feats, paths

print("Extracting train features and building memory bank...")
train_features, train_paths = extract_features(train_dataset)
num_memory = max(10, int(len(train_features)*0.1))
indices = np.random.choice(len(train_features), num_memory, replace=False)
memory_bank = train_features[indices]

nn_model = NearestNeighbors(n_neighbors=1)
nn_model.fit(memory_bank)
print("Memory bank ready.")

#test
print("Extracting test features and computing anomaly scores...")
test_features, test_paths = extract_features(test_dataset)
distances, _ = nn_model.kneighbors(test_features)
scores = distances.flatten()

for path, score in zip(test_paths, scores):
    print(f"{path} --> Anomaly Score: {score:.4f}")

#threshold
threshold = np.percentile(scores, 95)
anomaly_preds = [1 if s>threshold else 0 for s in scores]
for path, pred, score in zip(test_paths, anomaly_preds, scores):
    print(f"{path} --> Predicted: {'Anomaly' if pred else 'Normal'}, Score: {score:.4f}")
