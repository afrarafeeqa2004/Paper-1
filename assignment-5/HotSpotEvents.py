import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import deque

IMG_DIR = "/content/drive/MyDrive/images-2"
MASK_DIR = "/content/drive/MyDrive/masks-2"
MODEL_SAVE_PATH = "hotspot_alarm_final.pth"

IMG_SIZE = 256
BATCH_SIZE = 12
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
class SIRSTDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_list, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = img_list
        self.size = size
        self.mask_files = os.listdir(mask_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        #match image 'Misc_X.png' to mask 'Misc_X_pixels0.png' or similar
        base = os.path.splitext(img_name)[0]
        mask_name = next((f for f in self.mask_files if f.startswith(base) and
                         f.lower().endswith(('.png', '.bmp', '.jpg'))), None)

        if mask_name is None: raise FileNotFoundError(f"Mask missing for {img_name}")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load as Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        #binarize mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        #convert to Tensors
        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return img_t, mask_t

#attention model (for small targets)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)

class HotspotUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att = AttentionGate(32, 32, 16)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        up = self.up(x2)
        att = self.att(up, x1)
        return self.out(att) # output logits for stability

#loss function
def hybrid_loss(logits, targets):
    # pos_weight=100 means 'Hotspot' pixels are 100x more important than background
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]).to(DEVICE))(logits, targets)

    #dice Loss
    probs = torch.sigmoid(logits)
    smooth = 1e-6
    iflat = probs.view(-1)
    tflat = targets.view(-1)
    intersection = (iflat * tflat).sum()
    dice = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

    return bce + dice

def main():
    #split data
    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.png', '.jpg'))])
    train_list, test_list = train_test_split(all_files, test_size=0.2, random_state=42)

    train_loader = DataLoader(SIRSTDataset(IMG_DIR, MASK_DIR, train_list), batch_size=BATCH_SIZE, shuffle=True)

    #train
    model = HotspotUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting Training ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        l_sum = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = hybrid_loss(logits, masks)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {l_sum/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    #alarm evaluation
    model.eval()
    test_loader = DataLoader(SIRSTDataset(IMG_DIR, MASK_DIR, test_list), batch_size=1)
    persistence_window = deque(maxlen=5)

    #alarm threshold is raised back to 0.7 now that model is stronger
    THRESHOLD = 0.7

    print("\n--- ALARM RESULTS ---")
    for i, (img, _) in enumerate(test_loader):
        with torch.no_grad():
            logits = model(img.to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy().squeeze()

        conf = np.max(probs)
        detected = conf > THRESHOLD
        persistence_window.append(detected)

        # TRIGGER LOGIC: Must see hotspot in 3 out of 5 frames
        # (This keeps false alarms from noise extremely rare)
        if sum(persistence_window) >= 3:
            msg = "[!!! ALARM !!!]"
        else:
            msg = "OK"

        print(f"Frame {i} ({test_list[i]}): {msg} (Conf: {conf:.4f})")

if __name__ == "__main__":
    main()
