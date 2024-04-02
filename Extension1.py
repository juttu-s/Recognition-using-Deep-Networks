"""
Kirti Kshirsagar | Saikiran Juttu | 1st April 2024
This file represents one of the extensions for the task 2 of the assignment.
Here, we have loaded a pre-trained network, ResNet18, and visualized the filters of the first convolutional layer.
We have also applied the filters to the first training example image and visualized the filtered images.
"""

import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2

# Loading a pre-trained network, we chose to go with ResNet18
pretrained_model = models.resnet18(pretrained=True)

# Print the structure of the pre-trained model
print(pretrained_model)

# Get the weights of the first convolutional layer
first_layer_weights = pretrained_model.conv1.weight.data
print("Shape of the first layer weights:", first_layer_weights.shape)

# Get the weights of the second convolutional layer
second_conv_layer_weights = pretrained_model.layer1[0].conv1.weight.data
print("Shape of the second layer weights:", second_conv_layer_weights.shape)

# Visualize all the filters from the first convolutional layer
print("Filters:")
fig = plt.figure(figsize=(20, 16))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(first_layer_weights[i, 0].cpu(), cmap='viridis')
    plt.title(f'Filter {i}')
    plt.xticks([])
    plt.yticks([])
plt.show()

# Load the first training example image
train_data = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
first_image = train_data[0][0]  # the first image

# Apply the filters to the first training example image
filtered_images = []
with torch.no_grad():
    for i in range(first_layer_weights.shape[0]):
        filter = first_layer_weights[i, 0].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions to the filter
        filtered_image = torch.nn.functional.conv2d(first_image.unsqueeze(0), filter, padding=1)
        filtered_images.append(filtered_image.squeeze().numpy())

# Visualize the filtered images
fig = plt.figure(figsize=(20, 16))
for i in range(len(filtered_images)):
    plt.subplot(8, 8, i + 1)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title(f'Filtered Image {i}')
    plt.xticks([])
    plt.yticks([])
plt.show()
