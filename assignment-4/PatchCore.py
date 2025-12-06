import torch, glob, numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

device = "cuda" if torch.cuda.is_available() else "cpu"

train_folder = "/content/drive/MyDrive/xxxxx/train"
img_size = 256

#transform
tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

#load data
def load_images(folder):
    files = sorted([f for f in glob.glob(folder+"/*")])
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        imgs.append(tf(img))
    return torch.stack(imgs), files


#backbone (ResNet50 layer3)
model = models.resnet50(weights="IMAGENET1K_V1").to(device)
model.eval()

#hook layer
feat = None
def hook(m, i, o):
    global feat
    feat = o

model.layer3.register_forward_hook(hook)


#extract patch features
def get_features(img_batch):
    global feat
    with torch.no_grad():
        _ = model(img_batch.to(device))
    f = feat  # (B, C, H, W)
    B,C,H,W = f.shape
    f = f.view(B, C, H*W).permute(0,2,1)  # (B, Patches, C)
    return f.cpu().numpy(), (H,W)

train_imgs, train_paths = load_images(TRAIN_FOLDER)
train_feats, (h,w) = get_features(train_imgs)
train_feats = train_feats.reshape(-1, train_feats.shape[-1])  # (total_patches, C)
print("Train patch vectors:", train_feats.shape)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(train_feats)

#test
TEST_IMAGE = "/content/drive/MyDrive/xxxxx/test/003.png"

#load and transform
img_raw = Image.open(TEST_IMAGE).convert("RGB")
img = tf(img_raw).unsqueeze(0)  # add batch dimension

#extract patch features
test_feats, (h, w) = get_features(img)
test_feats = test_feats.reshape(-1, test_feats.shape[-1])  # (Patches, C)

#compute nearest-neighbor distances for all patches
distances, _ = nn.kneighbors(test_feats)

#anomaly score: maximum distance over all patches
anomaly_score = float(distances.max())
print("\nTest Image:\n", TEST_IMAGE)
print("Anomaly Score:", anomaly_score)

THRESHOLD = np.percentile(train_feats, 95)  # 95th percentile of train patches
if anomaly_score > THRESHOLD:
    print("DEFECT detected")
else:
    print("NORMAL image")
