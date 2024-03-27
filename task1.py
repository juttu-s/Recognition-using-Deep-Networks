import torch
import torchvision
import matplotlib.pyplot as plt

# Load MNIST test dataset
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Get the first six example digits and their corresponding labels
first_six_digits = mnist_test.data[:6]
labels = mnist_test.targets[:6]

# Plot the first six example digits
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)  # Create subplot grid of 2 rows and 3 columns
    plt.imshow(first_six_digits[i], cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.axis('off')  # Turn off axis
plt.tight_layout()
plt.show()

