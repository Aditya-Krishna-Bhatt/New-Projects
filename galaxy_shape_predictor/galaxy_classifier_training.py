# === 1. ENVIRONMENT WORKAROUNDS (DLL & SSL SECURITY BYPASS) ===
import os
import ctypes
import ssl
from importlib.util import find_spec

try:
    spec = find_spec("torch")
    if spec and spec.origin:
        dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        if os.path.exists(dll_path):
            ctypes.CDLL(os.path.normpath(dll_path))
except Exception:
    pass

ssl._create_default_https_context = ssl._create_unverified_context


# === 2. CORE MACHINE LEARNING SYSTEM EXECUTION ===
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from astroNN.datasets import load_galaxy10

print("Loading real Sloan Digital Sky Survey images from AstroNN...")
images, labels = load_galaxy10()

valid_mask = np.isin(labels, [1, 7, 9])
X_real = images[valid_mask]
y_raw = labels[valid_mask]

y_real = np.where(y_raw == 1, 1, 0)
X_real = np.transpose(X_real, (0, 3, 1, 2)).astype(np.float32) / 255.0
y_real = y_real.astype(np.int64)

# Calculate class weights to counter the dataset imbalance
num_spirals = np.sum(y_real == 0)
num_ellipticals = np.sum(y_real == 1)

# FIXED: Explicitly force the tensor to be a 32-bit Float using .float() to prevent Double mismatch
class_weights = torch.tensor([1.0, float(num_spirals) / num_ellipticals]).float()
print(f"Applying Imbalance Penalization Weights: {class_weights.numpy()}")

X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.20, random_state=42)

resize_transform = transforms.Resize((69, 69))

class RealGalaxyDataset(Dataset):
    def __init__(self, imgs, lbls):
        self.imgs = imgs
        self.lbls = lbls
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img_tensor = torch.tensor(self.imgs[idx])
        img_tensor = resize_transform(img_tensor)
        return img_tensor, torch.tensor(self.lbls[idx])

train_loader = DataLoader(RealGalaxyDataset(X_train, y_train), batch_size=64, shuffle=True)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealGalaxyCNN().to(device)

# The loss engine will now process the weights perfectly using the matching Float datatype
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nCommencing 10-Epoch training cycle over real astronomical inputs...")
model.train()
for epoch in range(10):
    loss_acc = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        loss_acc += loss.item()
    print(f"Epoch {epoch+1}/10 | Weighted Cross-Entropy Loss: {loss_acc/len(train_loader):.4f}")

torch.save(model.state_dict(), 'real_galaxy_weights.pth')
print("\nOptimized 10-Epoch model pipeline finalized successfully! Overwritten 'real_galaxy_weights.pth'")
