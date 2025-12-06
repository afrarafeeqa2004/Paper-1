#ViT with fine-tuning

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

base_dir = "./data_small/"
train_dir = base_dir + "train/"
test_dir = base_dir + "test/"

for split in ["train", "test"]:
    for cls in range(10):
        os.makedirs(f"{base_dir}{split}/{cls}", exist_ok=True)

#load dataset
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True, transform=transform)

#save images into folders
print("Saving CIFAR-10 images. Please wait...")

for idx, (img, label) in enumerate(trainset):
    save_image(img, f"{train_dir}/{label}/{idx}.png")

for idx, (img, label) in enumerate(testset):
    save_image(img, f"{test_dir}/{label}/{idx}.png")

print("Dataset ready: data_small/train and data_small/test")

import torch
from torch import nn, autocast
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

#transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("./data_small/train", transform=train_transform)
test_data = datasets.ImageFolder("./data_small/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

#load ViT-tiny pretrained
vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
vit.head = nn.Linear(vit.head.in_features, len(train_data.classes))
vit = vit.to(device)

#optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit.parameters(), lr=1e-4)

#training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

EPOCHS = 5

for epoch in range(EPOCHS):
    vit.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed precision
            outputs = vit(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

#evaluate
vit.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vit(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
vit_acc=correct/total
print(f"ViT-Tiny Test Accuracy: {vit_acc:.4f}")

#test
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

img_path = "./data_small/test/0/10.png"

#define transforms (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#load and transform image
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

vit.eval()
with torch.no_grad():
    output = vit(img_tensor)
    pred_idx = torch.argmax(output, dim=1).item()

#get true label from folder structure
true_idx = int(img_path.split('/')[-2])
classes = train_data.classes

plt.imshow(img)
plt.title(f"True: {classes[true_idx]} | Pred: {classes[pred_idx]}")
plt.axis('off')
plt.show()

#CNN without fine-tuning

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#transform
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

#load dataset
train_data = datasets.ImageFolder("./data_small/train", transform=train_transform)
test_data = datasets.ImageFolder("./data_small/test", transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

#CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

cnn = SimpleCNN(num_classes=len(train_data.classes)).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

#training
EPOCHS = 5
for epoch in range(EPOCHS):
    cnn.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

#evaluate
cnn.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
cnn_acc=correct/total
print(f"CNN Test Accuracy: {cnn_acc:.4f}")

#predict
def predict_single_image_cnn(img_path, model, classes, device):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    plt.imshow(img)
    plt.title(f"Predicted: {classes[pred_idx]}")
    plt.axis('off')
    plt.show()

    return classes[pred_idx]

predict_single_image_cnn("./data_small/test/0/10.png", cnn, train_data.classes, device)

import matplotlib.pyplot as plt

#accuracy values
models = ['vit', 'cnn']
accuracies = [vit_acc, cnn_acc]

plt.figure(figsize=(6,5))
plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Comparison of Test Accuracy: ViT vs CNN')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center', fontsize=12)
plt.show()
