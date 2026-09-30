import os
import ctypes
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from astroNN.datasets import load_galaxy10

# 1. WINDOWS DLL BYPASS PATHS
try:
    from importlib.util import find_spec
    spec = find_spec("torch")
    if spec and spec.origin:
        dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        if os.path.exists(dll_path):
            ctypes.CDLL(os.path.normpath(dll_path))
except Exception:
    pass

# 2. RE-ESTABLISH THE CNN BLUEPRINT
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

# 3. LOAD DATASET AND WEIGHTS
print("Loading data arrays for visual plotting...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealGalaxyCNN().to(device)
model.load_state_dict(torch.load('real_galaxy_weights.pth', map_location=device))
model.eval()

images, labels = load_galaxy10()
valid_mask = np.isin(labels, [1, 7, 9])
X_real = np.transpose(images[valid_mask], (0, 3, 1, 2)).astype(np.float32) / 255.0
y_real = np.where(labels[valid_mask] == 1, 1, 0).astype(np.int64)

_, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.20, random_state=42)
resize_transform = transforms.Resize((69, 69))

# 4. PLOT A RANDOM SAMPLE GRID OF REAL GALAXIES WITH AI PREDICTIONS
print("\nGenerating 'galaxy_predictions_grid.png'...")
sample_indices = np.random.choice(len(X_test), size=5, replace=False)
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
class_names = ['Spiral', 'Elliptical']

for i, idx in enumerate(sample_indices):
    img_raw = X_test[idx]
    true_label = y_test[idx]
    
    # Process image for model inference entry
    img_tensor = resize_transform(torch.tensor(img_raw)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    
    pred_label = np.argmax(probabilities)
    confidence = probabilities[pred_label]
    
    # Transpose back to standard plotting orientation (H, W, C)
    img_display = np.transpose(img_raw, (1, 2, 0))
    img_display = np.clip(img_display, 0, 1) # Safeguard pixels
    
    axes[i].imshow(img_display)
    title_color = 'green' if pred_label == true_label else 'red'
    axes[i].set_title(f"True: {class_names[true_label]}\nAI: {class_names[pred_label]}\nConf: {confidence:.1%}", color=title_color)
    axes[i].axis('off')

plt.suptitle("Testing Trained Model Predictions on Real Sky Survey Imagery", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('galaxy_predictions_grid.png', dpi=300)
plt.close()

# 5. GENERATE THE MATPLOTLIB CONFUSION MATRIX CHART
print("Generating 'real_confusion_matrix.png'...")
all_preds = []
for img in X_test:
    img_t = resize_transform(torch.tensor(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
        pred = torch.argmax(out, dim=1).cpu().item()
        all_preds.append(pred)

cm = confusion_matrix(y_test, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Galactic Class')
plt.ylabel('True Galactic Class')
plt.title('Galaxy Morphology Confusion Matrix (Real Data)')
plt.tight_layout()
plt.savefig('real_confusion_matrix.png', dpi=300)
plt.close()

print("\nSuccess! Generated 'galaxy_predictions_grid.png' and 'real_confusion_matrix.png' inside your folder!")
