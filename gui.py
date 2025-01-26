import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
import torch
from torchvision import transforms
from PIL import Image, ImageTk
import cv2
import numpy as np
from torch import nn

# Define the custom CNN model (same as used for training)
class SleepAwakeCNN(nn.Module):
    def __init__(self):
        super(SleepAwakeCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Output: awake or asleep
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load the trained model
model = SleepAwakeCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('Drowsiness_model.pth', map_location=device))
model.to(device)
model.eval()

# Transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to process image and make predictions
def predict(image):
    image = Image.open(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = "Asleep" if predicted.item() == 1 else "Awake"
    
    # Return the predicted label (could be expanded to include age prediction if available)
    return label

# Function to open file dialog and process the selected image/video
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        label = predict(file_path)
        display_image(file_path)
        messagebox.showinfo("Prediction", f"The person is {label}.")

# Function to display the selected image in the Tkinter window
def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((400, 400))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

# Create the main GUI window
root = tk.Tk()
root.title("Sleep/Awake Detection with Model")
root.geometry("600x600")

# Label to show image preview
image_label = Label(root)
image_label.pack(pady=10)

# Button to upload an image
upload_button = Button(root, text="Upload Image/Video", command=upload_image)
upload_button.pack(pady=20)

# Run the GUI
root.mainloop()
