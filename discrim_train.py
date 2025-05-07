import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
from discrimin_model import Discriminative_model  # Import your model
import matplotlib.pyplot as plt

class Data_MRI(Dataset):
    def __init__(self, mri_dir, ct_dir, transform=None):
        self.mri_dir = mri_dir
        self.ct_dir = ct_dir
        self.mri_images = sorted(os.listdir(mri_dir))
        self.ct_images = sorted(os.listdir(ct_dir))
        self.transform = transform

    def __len__(self):
        return len(self.mri_images)

    def __getitem__(self, idx):
        mri_path = os.path.join(self.mri_dir, self.mri_images[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_images[idx])

        mri_image = Image.open(mri_path).convert("L")  
        ct_image = Image.open(ct_path).convert("L")  

        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)

        return mri_image, ct_image

    def pad_image(self, image):
        width, height = image.size
        new_width = (width + 31) // 32 * 32
        new_height = (height + 31) // 32 * 32
        padded_image = Image.new("L", (new_width, new_height))
        padded_image.paste(image, (0, 0))
        return padded_image

# Define Dataset Paths
mri_train_dir = "C:/Shreyas/AI courseowkr/png_dataset/train/mri"
ct_train_dir = "C:/Shreyas/AI courseowkr/png_dataset/train/ct"
mri_val_dir = "C:/Shreyas/AI courseowkr/png_dataset/val/mri"
ct_val_dir = "C:/Shreyas/AI courseowkr/png_dataset/val/ct"

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define paths
CHECKPOINT_PATH = "Discriminative_model_checkpoint.pth"
LOSS_LOG_PATH = "loss_log.txt"

# Create Datasets and Dataloaders
train_dataset = Data_MRI(mri_train_dir, ct_train_dir, transform)
val_dataset = Data_MRI(mri_val_dir, ct_val_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Discriminative_model().to(device)

# Define Loss and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load checkpoint if exists
start_epoch = 0
if os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

# Training Loop
def train_model(num_epochs=200):
    global best_val_loss, epochs_no_improve
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for mri_images, ct_images in train_loader:
            mri_images, ct_images = mri_images.to(device), ct_images.to(device)

            optimizer.zero_grad()
            outputs = model(mri_images)
            loss = criterion(outputs, ct_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mri_images, ct_images in val_loader:
                mri_images, ct_images = mri_images.to(device), ct_images.to(device)
                outputs = model(mri_images)
                loss = criterion(outputs, ct_images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), "Discriminative_model_best.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    # Save the final model
    torch.save(model.state_dict(), "Discriminative_model_final.pth")
    print("Training Complete! Model saved as Discriminative_model_final.pth")

    # Save loss values to file
    with open(LOSS_LOG_PATH, 'w') as f:
        for t_loss, v_loss in zip(train_losses, val_losses):
            f.write(f"{t_loss},{v_loss}\n")

if __name__ == "__main__":
    train_model(num_epochs=200)