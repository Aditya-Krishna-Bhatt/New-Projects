# === 1. ENVIRONMENT WORKAROUNDS ===
import os
import ctypes
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import classification_report

try:
    spec = find_spec("torch")
    if spec and spec.origin:
        dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        if os.path.exists(dll_path):
            ctypes.CDLL(os.path.normpath(dll_path))
except Exception:
    pass

# === 2. MODEL CORE & EVALUATION ===
class RealGalaxyCNN(nn.Module):
    def __init__(self):
        super(RealGalaxyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 17 * 17, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Re-instantiate architecture and map trained weights natively
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealGalaxyCNN().to(device)
model.load_state_dict(torch.load('real_galaxy_weights.pth', map_location=device))
model.eval()

# Load the local test arrays from memory to score them
from astroNN.datasets import load_galaxy10
_, labels = load_galaxy10()
valid_mask = np.isin(labels, [1, 7, 9])
images = load_galaxy10()[0][valid_mask]
labels = labels[valid_mask]
y_real = np.where(labels == 1, 1, 0)
X_real = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0

# Isolate the identical 20% test slice using the same random seed
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.20, random_state=42)

resize_transform = transforms.Resize((69, 69))
class TestDataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = resize_transform(torch.tensor(self.imgs[idx]))
        return img, torch.tensor(self.lbls[idx])

test_loader = DataLoader(TestDataset(X_test, y_test), batch_size=64, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        outputs = model(imgs.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.numpy())

print("\n" + "="*50)
print("   🌌 SLOAN DIGITAL SKY SURVEY - TEST RESULTS")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=['Spiral', 'Elliptical']))
