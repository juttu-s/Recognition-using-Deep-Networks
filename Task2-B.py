import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import matplotlib.pyplot as plt

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