"""
Saikiran Juttu | Kirti Kshirsagar | Project-5 | 4th April 2024 | Spring 2024
This file represents Task3 of the assignment.
Here, we are transfering the neural network model and replacing the last layer to detect the greek letters, We trained this 
model on train data set and plotted the training error graph, later we test this model on handwritten greek laters and presented results with the predicted labels.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys

# Class definitions
class MyNetwork(nn.Module):
    """Define a neural network model."""
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Define layers and operations in the network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)  # This is the last layer for MNIST

    def forward(self, x):
        """Computes a forward pass for the network."""
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Freeze parameters for the whole network
def freeze_parameters(network):
    for param in network.parameters():
        param.requires_grad = False

# Modify the last layer for Greek letters
def modify_last_layer(network):
    network.fc2 = nn.Linear(50, 3)  # Replace the last layer with 3 nodes for Greek letters

# Load pre-trained weights
def load_pretrained_weights(network, model_path):
    network.load_state_dict(torch.load(model_path))

# Greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Main function
def main(argv):

    # Load pre-trained MNIST network
    network = MyNetwork()

    # Load pre-trained weights
    model_path = './results/model.pth'
    load_pretrained_weights(network, model_path)

    # Freeze network weights
    freeze_parameters(network)

    # Modify last layer for Greek letters
    modify_last_layer(network)

    # Define optimizer
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

    # Define Greek data loader
    training_set_path = '/home/sakiran/Recognition-using-Deep-Networks/greek_train'
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              GreekTransform(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ])),
        batch_size=5,
        shuffle=True)

    # Training loop
    n_epochs = 120
    training_errors = []  # List to store training errors
    for epoch in range(n_epochs):
        epoch_error = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_error += loss.item()  # Accumulate loss for this epoch
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)
            print('Epoch {} Batch {}: Loss: {:.6f}'.format(epoch, batch_idx, loss.item()))
        training_errors.append(epoch_error / len(greek_train))  # Average loss for this epoch
        training_accuracy = correct_predictions / total_samples
        print('Epoch {}: Training Accuracy: {:.2f}%'.format(epoch, training_accuracy * 100))

    # Plot the training error
    plt.plot(training_errors)
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.title('Training Error vs. Epoch')
    plt.show()

    # Printout of the modified network
    print(network)

    # Define Greek test data loader
    test_set_path = '/home/sakiran/Recognition-using-Deep-Networks/test_greek'
    greek_test = torchvision.datasets.ImageFolder(test_set_path,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.Resize((128, 128)),
                                                       torchvision.transforms.ToTensor(),
                                                       GreekTransform(),
                                                       torchvision.transforms.Normalize(
                                                           (0.1307,), (0.3081,))
                                                   ]))

    # Testing loop
    correct_predictions = 0
    total_samples = 0
    predictions = []

    class_labels = {0: 'Alpha', 1: 'Beta', 2: 'Gamma'}

    for data, target in greek_test:
        data = data.unsqueeze(0)  # Add batch dimension
        output = network(data)
        _, predicted = torch.max(output, 1)
        total_samples += 1
        correct_predictions += (predicted == target).item()
        predictions.append((data.squeeze().numpy(), target, class_labels[predicted.item()]))

    # Calculate test accuracy
    test_accuracy = correct_predictions / total_samples
    print('Test Accuracy: {:.2f}%'.format(test_accuracy * 100))


    # Plot the images with predicted labels
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 6))
    for i, (data, target, predicted) in enumerate(predictions):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(data, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Predicted: {predicted}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)