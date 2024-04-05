"""
Kirti Kshirsagar | Saikiran Juttu | 1st April 2024
This code is written for the task 2 of the assignment. The code is written in Python and uses PyTorch library to load 
the trained model and analyze the first layer of the model. The code also shows the effect of the filters on the first 
training example image.
"""
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2

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

# Load the trained model
model = Net()
model.load_state_dict(torch.load('model.pth'))

# Task A: Analyze the first layer
# Print the model structure
print(model)

# Analyze the first layer
first_layer_weights = model.conv1.weight.data
print("Shape of the first layer weights:", first_layer_weights.shape)

# Print the filter weights and their shape
for i in range(10):
    print(f"Filter {i} shape:", first_layer_weights[i, 0].shape)
    print(f"Filter {i} weights:\n", first_layer_weights[i, 0])
    print()

# Visualize the ten filters
fig = plt.figure(figsize=(10, 8))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(first_layer_weights[i, 0].cpu(), cmap='viridis')
    plt.title(f'Filter {i}')
    plt.xticks([])
    plt.yticks([])
plt.show()

# Task B: Show the effect of the filters on the first training example image
# Load the first training example image
train_data = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())
first_image = train_data[0][0]  # Get the first image

# Get the weights of the first layer
with torch.no_grad():
    first_layer_weights = model.conv1.weight.data

# Set up the subplots
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 10))

# Plot filters and filtered images
for i in range(10):
    col = i // 5
    row = i % 5

    # Plot filter
    axes[row, col * 2].imshow(first_layer_weights[i, 0].cpu().numpy(), cmap='gray')
    axes[row, col * 2].set_title(f'Filter {i}')
    axes[row, col * 2].axis('off')

    # Plot filtered image
    filter = first_layer_weights[i, 0].cpu().numpy()
    filtered_image = cv2.filter2D(first_image.squeeze().numpy(), -1, filter)
    axes[row, col * 2 + 1].imshow(filtered_image, cmap='gray')
    axes[row, col * 2 + 1].set_title(f'Filtered Image {i}')
    axes[row, col * 2 + 1].axis('off')

plt.tight_layout()
plt.show()