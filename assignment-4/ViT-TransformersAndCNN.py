#ViT-transformers
!pip install -q keras-cv keras-hub tensorflow matplotlib numpy

!pip install --upgrade keras-hub

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras_cv
from tensorflow import keras
#from keras_cv.models.backbones import ViTBackbone
#from keras_cv.models.classification import ViTClassifier
from keras_hub.models import ViTBackbone, ViTImageClassifier

#load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

#normalize images to [0,1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

num_classes = 10
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("Training data shape:", x_train.shape)
print("Testing data shape :", x_test.shape)

from keras_hub.models import Backbone

#build vision transformer model
vit_backbone = Backbone.from_preset("vit_base_patch16_224_imagenet")

#freeze backbone
vit_backbone.trainable = False

#add classification head for CIFAR-10
model = keras.Sequential([
    keras.layers.Resizing(224, 224),
    vit_backbone,
    keras.layers.GlobalAveragePooling1D(), #add Global Average Pooling layer
    keras.layers.Dense(num_classes, activation="softmax")
])

#compile and train briefly
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    validation_split=0.1,
                    epochs=2,       
                    batch_size=64,
                    verbose=1)

#evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
vit_acc=test_acc
print(f"\n Test accuracy: {vit_acc:.4f}")

#predict and display results
pred = np.argmax(model.predict(x_test[:10]), axis=1)

plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i])
    plt.title(f"Pred: {class_names[pred[i]]}")
    plt.axis('off')
plt.show()

#CNN

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#transform
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

#load dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

#CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
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

cnn = SimpleCNN().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

#training
EPOCHS = 10
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

cnn_acc = correct / total
print(f"CNN Test Accuracy on CIFAR-10: {cnn_acc:.4f}")

import matplotlib.pyplot as plt

#accuracy values
models = ['model', 'cnn']
accuracies = [vit_acc, cnn_acc]

plt.figure(figsize=(6,5))
plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Comparison of Test Accuracy: ViT vs CNN')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center', fontsize=12)
plt.show()
