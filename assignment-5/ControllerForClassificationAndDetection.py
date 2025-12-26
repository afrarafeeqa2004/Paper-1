!pip install torch torchvision opencv-python numpy pandas scikit-learn

import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import deque

BASE_PATH = "/content/drive/MyDrive/hazelnut"
MODEL_A_SAVE = "hazelnut_classifier.pth"
MODEL_B_SAVE = "hazelnut_detector.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

#data mapping
def map_hazelnut_data(root):
    data = []
    # 1. Map GOOD (Label 0)
    for sub in ["train/good", "test/good"]:
        folder = os.path.join(root, sub)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".png"):
                    data.append({"img": os.path.join(folder, f), "mask": None, "label": 0, "type": "good"})

    # 2. Map DEFECTS (Label 1)
    test_root = os.path.join(root, "test")
    gt_root = os.path.join(root, "ground_truth")
    defect_types = [d for d in os.listdir(test_root) if d != "good"]

    for d_type in defect_types:
        img_dir = os.path.join(test_root, d_type)
        mask_dir = os.path.join(gt_root, d_type)
        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if f.endswith(".png"):
                    mask_name = f.replace(".png", "_mask.png")
                    mask_path = os.path.join(mask_dir, mask_name)
                    data.append({
                        "img": os.path.join(img_dir, f),
                        "mask": mask_path if os.path.exists(mask_path) else None,
                        "label": 1,
                        "type": d_type
                    })
    return pd.DataFrame(data)

class HazelnutDataset(Dataset):
    def __init__(self, df, size=256):
        self.df = df
        self.size = size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['img'])
        img = cv2.resize(img, (self.size, self.size))

        #binary Mask
        if row['mask']:
            mask = cv2.imread(row['mask'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.size, self.size))
        else:
            mask = np.zeros((self.size, self.size), dtype=np.uint8)

        img_t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        return img_t, mask_t, torch.tensor([row['label']]).float()

# 4. MODELS (A: Classify, B: Detect)
class ModelA_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class ModelB_Detector(nn.Module): #simple U-Net style
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 1, 2, stride=2), nn.Sigmoid())
    def forward(self, x): return self.dec(self.enc(x))

#the sequential controller
class SequentialController:
    def __init__(self, model_a, model_b):
        self.model_a = model_a.to(DEVICE).eval()
        self.model_b = model_b.to(DEVICE).eval()
        self.logs = []

    def run_inference(self, test_df):
        print(f"Running Sequential Inspection on {len(test_df)} test items...")
        dataset = HazelnutDataset(test_df)

        for i in range(len(test_df)):
            img_t, _, actual_label = dataset[i]
            img_t = img_t.unsqueeze(0).to(DEVICE)

            entry = {"File": os.path.basename(test_df.iloc[i]['img']),
                     "Actual": "Anomaly" if test_df.iloc[i]['label']==1 else "Normal"}

            #stage-1: classify
            t0 = time.perf_counter()
            with torch.no_grad():
                is_anomaly = self.model_a(img_t).item() > 0.5
            entry["Time_A_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            #stage-2: detect(gated)
            entry["Time_B_ms"] = 0.0
            if is_anomaly:
                t1 = time.perf_counter()
                with torch.no_grad():
                    _ = self.model_b(img_t)
                entry["Time_B_ms"] = round((time.perf_counter() - t1) * 1000, 2)
                entry["Decision"] = "ALERT (Localized)"
            else:
                entry["Decision"] = "PASS"

            entry["Total_Latency_ms"] = entry["Time_A_ms"] + entry["Time_B_ms"]
            self.logs.append(entry)

    def get_report(self):
        return pd.DataFrame(self.logs)

if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print("Path not found. Set BASE_PATH correctly.")
    else:
        #prepare data
        df = map_hazelnut_data(BASE_PATH)
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        #initialize models
        m_a = ModelA_Classifier().to(DEVICE)
        m_b = ModelB_Detector().to(DEVICE)

        # 3. Simulate Training (Normally you would run your train loops here)
        print("Training Models (Placeholder)...")
        torch.save(m_a.state_dict(), MODEL_A_SAVE)
        torch.save(m_b.state_dict(), MODEL_B_SAVE)

        #run controller
        controller = SequentialController(m_a, m_b)
        controller.run_inference(test_df)

        #generate combined report
        report = controller.get_report()
        print("\n" + "="*80)
        print("SEQUENTIAL HAZELNUT INSPECTION REPORT")
        print("="*80)
        print(report.to_string(index=False))
        print("="*80)
        print(f"Avg Latency: {report['Total_Latency_ms'].mean():.2f} ms")
