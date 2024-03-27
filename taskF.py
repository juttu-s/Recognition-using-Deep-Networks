import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Define the network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# Load the pre-trained model
model = Net()
model.load_state_dict(torch.load('./results/model.pth'))
model.eval()  # Set model to evaluation mode

# Load and preprocess the images
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((28, 28))
    image = Image.fromarray(255 - np.array(image))
    # Convert image to tensor and normalize
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    # Normalize the image tensor
    # image_tensor = (image_tensor - 0.1307) / 0.3081
    return image_tensor

# Test the network on new inputs
def test_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = preprocess_image(image_path)
            # Display the image
            plt.imshow(image.squeeze().numpy(), cmap='gray')
            plt.axis('off')
            plt.show()
            # Run the image through the network
            with torch.no_grad():
                output = model(image)
                probabilities = torch.exp(output)
                predicted_label = torch.argmax(output, dim=1).item()
                print(f"Predicted Label: {predicted_label}")
                print("Probabilities:")
                for i, prob in enumerate(probabilities.squeeze().tolist()):
                    print(f"  Digit {i}: {prob:.4f}")

# Path to the folder containing images of handwritten digits
folder_path = "./sample"

# Test the network on images from the folder
test_images_from_folder(folder_path)
