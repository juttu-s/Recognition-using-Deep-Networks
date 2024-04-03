"""
Kirti Kshirsagar | Saikiran Juttu | 1st April 2024
This code file represents second extension. 
In this code, we have replaced the first layer of the network with Gabor filters and 
trained the network on the MNIST dataset, having the first layer constant.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import torch.nn.functional as F

# Gabor filter function
def gabor_filter(size, wavelength, orientation):
    sigma = 2 * np.pi
    gamma = 0.5
    psi = 0
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            x_prime = x * np.cos(orientation) + y * np.sin(orientation)
            y_prime = -x * np.sin(orientation) + y * np.cos(orientation)
            kernel[x, y] = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2)) * \
                           np.cos(2 * np.pi * x_prime / wavelength + psi)
    return kernel / np.sum(kernel)

# New network class with the first layer replaced by Gabor filters
class MNISTGaborNet(nn.Module):
    def __init__(self, gabor_filters):
        super(MNISTGaborNet, self).__init__()
        self.conv1 = nn.Conv2d(1, len(gabor_filters), kernel_size=gabor_filters[0].shape[0], bias=False)
        for i, filter in enumerate(gabor_filters):
            self.conv1.weight.data[i, 0, :, :] = torch.FloatTensor(filter)

        self.conv2 = nn.Conv2d(len(gabor_filters), 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# Defining Gabor filter bank
num_filters = 2
gabor_filters = [gabor_filter(size=5, wavelength=2, orientation=i * np.pi / num_filters) for i in range(num_filters)]

# Replace the first layer of the network with Gabor filters
model = MNISTGaborNet(gabor_filters)

# Freeze the weights of the first layer
model.conv1.weight.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9)

# Train the network
num_epochs = 5
train_losses = []  # Store training losses for plotting
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            train_losses.append(running_loss / 100)  # Append current loss for plotting
            running_loss = 0.0

print('Finished Training')

# Plot training loss over epochs
plt.plot(np.linspace(0, num_epochs, len(train_losses)), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()


# Test the network
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
