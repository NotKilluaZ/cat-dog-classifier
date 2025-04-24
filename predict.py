import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from main import CNN # Import Network Class from main.py

# Setup (same as main.py --> ie. check for GPU else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load our trained model
model = CNN().to(device)
model.load_state_dict(torch.load("model/cat_dog_model.pth"))
model.eval()

# Load and Preprocess Image
img_path = "test_img/bambi_1.jpeg" # Edit this line to change Image used for test
# Opens Image
img = Image.open(img_path)

# Transform Image to Resize
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

img_tensor = transform(img).unsqueeze(0).to(device)

# Predict Image Cat or Dog
with torch.no_grad():
    # Send Image into Trained Model
    output = model(img_tensor)
    probability = torch.sigmoid(output) # Manually Apply Sigmoid (took it out of main.py)
    # Get Prediction Based on Output
    prediction = "Dog" if probability.item() > 0.5 else "Cat 🐱"

# Show Image & Prediction
plt.imshow(img)
plt.title(f"Prediction: {prediction}")
plt.axis("off")
plt.show()