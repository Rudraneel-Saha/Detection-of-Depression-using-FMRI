# model/train.py
import torch

from torch.utils.data import DataLoader, random_split
#from models.model import DepressionCNN
#from utils import FMRI_Dataset, get_file_label_list

 #THIS IS THE MODEL

#import torch
import torch.nn as nn

class DepressionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

#THIS IS PREPROCESS

# model/preprocess.py
import nibabel as nib
import numpy as np
#import torch
import torch.nn.functional as F
import os
from skimage.transform import resize

def preprocess_nifti(nifti_path, target_size=(64, 64)):
    img = nib.load(nifti_path).get_fdata()
    # Example: select middle slice from the third axis (update if needed)
    slice_idx = img.shape[2] // 2
    data_slice = img[:, :, slice_idx]
    # Min-max normalization
    data_slice = (data_slice - np.min(data_slice)) / (np.ptp(data_slice) + 1e-7)
    # Resize
    data_slice = resize(data_slice, target_size, preserve_range=True)
    # Add channel for CNN [1, H, W]
    tensor = torch.tensor(data_slice, dtype=torch.float32).unsqueeze(0)
    return tensor


#THIS IS PREDICT

#model/predict.py
import torch
#from models.model import DepressionCNN
#from preprocess import preprocess_nifti

model = DepressionCNN()
model.load_state_dict(torch.load('depression_cnn.pt', map_location='cpu'))
model.eval()

def predict_nifti(nifti_path):
    tensor = preprocess_nifti(nifti_path).unsqueeze(0)  # [B, 1, 64, 64]
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1).item()
    return pred  # 0: control, 1: depressed


#utils
    
#import torch
from torch.utils.data import Dataset
#import os
#from preprocess import preprocess_nifti # update if using different preprocess location

# Your data root path
root_dir = "C:/Users/Rudraneel_Saha/OneDrive/Desktop/Major Project 7th sem"
def get_file_label_list(root_dir):
    samples = []
    for label_dir, label in [('control', 0), ('depressed', 1)]:
        folder = os.path.join(root_dir, label_dir)
        if not os.path.isdir(folder):
            continue  # skip missing directories
        for f in os.listdir(folder):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                samples.append((os.path.join(folder, f), label))
    return samples

class FMRI_Dataset(Dataset):
    def __init__(self, file_label_list):
        self.samples = file_label_list
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        tensor = preprocess_nifti(file_path)  # assumes single slice for each subject, update for 3D if needed
        return tensor, torch.tensor(label, dtype=torch.long)

# Create labeled file list using the correct path
file_label_list = get_file_label_list(root_dir)
# Now use file_label_list for FMRI_Dataset or split into train/test as shown earlier
 



# Paths
# Right!
data_dir = r"C:\Users\Rudraneel_Saha\OneDrive\Desktop\Major Project 7th sem\Dataset"
all_samples = get_file_label_list(data_dir)
split_idx = int(0.8 * len(all_samples))
train_list = all_samples[:split_idx]
test_list = all_samples[split_idx:]
train_ds = FMRI_Dataset(train_list)
test_ds = FMRI_Dataset(test_list)
if len(train_list) == 0:
    print("Training set is empty! Check your data organization.")
    exit()
if len(test_list) == 0:
    print("Test set is empty! Check your data organization.")
    exit()
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

model = DepressionCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(10):
        for x, y in train_loader:
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
        print(f'Epoch {epoch+1} done')
    torch.save(model.state_dict(), 'depression_cnn.pt')

def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            pred_labels = preds.argmax(1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    print('Test Accuracy:', correct/total)
    print('Correct - ', correct)
    print ('Total - ', total)

if __name__ == '__main__':
    train()
    test()
