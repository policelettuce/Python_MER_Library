import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

torch.cuda.empty_cache()

class MelSpectrogramDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.annotations.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert("RGB")
        valence = self.annotations.iloc[idx, 1]
        arousal = self.annotations.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([valence, arousal], dtype=torch.float32)


# Define the transformation
transform = transforms.Compose([
    transforms.Resize((400, 1000)),  # Resize to match the expected input size
    transforms.ToTensor()
])

# Dataset and DataLoader
dataset = MelSpectrogramDataset(
    annotations_file="dataset/pmemo_annotations_normalized.csv",
    img_dir="dataset/mel_spectrograms",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load the pretrained ConvNeXt model
model = timm.create_model('convnext_base', pretrained=True, num_classes=2)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


model_save_path = "convnext_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
print("Finished Training")
