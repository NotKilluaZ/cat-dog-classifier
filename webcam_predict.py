import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from main import CNN

# Setup
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = CNN().to(device)
model.load_state_dict(torch.load("model/cat_dog_model.pth"))
model.eval()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Open Webcam
cap = cv2.VideoCapture(0) # 0 is default webcam

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert Frame to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # Transform & Predict Output
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output) # Manually Apply Sigmoid
        confidence = probability.item()
        prediction = "Dog" if probability.item() > 0.5 else "Cat"
        # Confidence Reading
        confidence_pct = int(confidence * 100 if confidence > 0.5 else (1 - confidence) * 100)

    # === Draw prediction text ===
    label = f"{prediction} ({confidence_pct}%)"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw confidence bar
    bar_x, bar_y = 10, 70
    bar_width = 300
    bar_height = 20
    filled_width = int(bar_width * (confidence if confidence > 0.5 else 1 - confidence))

    # Draw outline
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    # Draw filled portion
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)

    # Show Frame
    cv2.imshow('Cat vs Dog Classifier', frame)

    # Press 'q' To Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release & Close
cap.release()
cv2.destroyAllWindows()