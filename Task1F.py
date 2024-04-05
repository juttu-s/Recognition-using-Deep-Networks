"""
Saikiran Juttu | Kirti Kshirsagar | Project-5 | 28 March 2024 | Spring 2024
This file represents Task1- F of the assignment.
Here, we are loading the pre trained neural network model and testing it on handwritten digits data.
"""

# import statements
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# class definitions
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# useful functions with a comment for each function
def preprocess_image(image_path):
    """
    Preprocesses an image for input to the neural network.
    """
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((28, 28))
    image = Image.fromarray(255 - np.array(image))
    # Convert image to tensor and normalize
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    # Normalize the image tensor
    image_tensor = (image_tensor - 0.1307) / 0.3081
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image_tensor

# main function (yes, it needs a comment too)
def main(argv):
    """
    Main function to execute the script.
    """
     # Load the pre-trained model
    model = Net()
    model.load_state_dict(torch.load('./results/model.pth'))
    model.eval()  # Set model to evaluation mode
    # Path to the folder containing images of handwritten digits
    folder_path = "./handwritten_digits"

    # Test the network on images from the folder
    images = []
    predictions = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = preprocess_image(image_path)
            images.append(image)
            # Run the image through the network
            with torch.no_grad():
                output = model(image)
                predicted_label = torch.argmax(output, dim=1).item()
                predictions.append(predicted_label)

    # Plot the first 9 digits with predictions
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].squeeze().numpy(), cmap='gray')  
        ax.set_title(f'Prediction: {predictions[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)

