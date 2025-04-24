import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Utilize GPU if available; else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformations
# Updated training image transforms. Rotating images randomly, different crop sizes, and flipping
# Helps train model better for realistic circumstances (ex. bad lighting // angle)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

# This part ensures all images are equal size and format
val_transform = transforms.Compose([
    transforms.Resize((150, 150)), # Resize images to consistent size 150 x 150
    transforms.ToTensor(),         # Convert images to tensors (3D Matrix)
])

# Load Dataset
# Here we load images and assign them labels based on folder names
train_dataset = datasets.ImageFolder("data/train", transform = train_transform)
val_dataset = datasets.ImageFolder("data/val", transform = val_transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32)

# Define Simple Convolutional Neural Network Model (CNN)
# Training Loop learns patterns from the images to better classify them (Machine Learning)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding = 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding = 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding = 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# Wrapper condition so that the model is not trained every time we call it in other files
if __name__ == "__main__":

    model = CNN().to(device)

    # Load existing trained model if available
    model_path = "model/cat_dog_model.pth"
    if os.path.exists(model_path):
        print("🔄 Loading existing model...")
        model.load_state_dict(torch.load(model_path))

    # Loss Function and Optimizer
    # BCELoss is a loss function that measures how well the model is classifying cats or dogs
    loss_fn = nn.BCEWithLogitsLoss()  # Updated BCE Loss function for numeric stability
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    # Each EPOCH is an iteration of the training loop (ex. range of 3 will train the model 5 times)
    for epoch in range(1):
        # The function that actually trains the model on given images
        model.train()

        # Initialize tracking variables
        total_loss = 0
        correct = 0
        total = 0

        # Iterate through each image
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device) # Reshape for BCELoss

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Optimizer helps improve our model after each training round
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += ((outputs > 0.5).float() == labels).sum().item()
            total += labels.size(0)

        # Compare correctly labeled images with total images labeled (* 100 for %)
        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save Trained Model
    # Saving the model allows us to re-use it later without retraining from scratch
    os.makedirs("model", exist_ok = True)
    torch.save(model.state_dict(), "model/cat_dog_model.pth")