"""
Saikiran Juttu | Kirti Kshirsagar | 1st April 2024
This file represents Task4 (Design your own experiment) of the assignment.
Here, we are determining the range of options in size of the convolution filters, Whether to add another dropout layer after the fully connected layer,
dropout rates of the Dropout layer, The number of epochs of training, The batch size while training, to explore and printing out the top 5 combinations.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Class definitions
class MyNetwork(nn.Module):
    """Define a neural network model."""
    def __init__(self, conv_filter_size, pool_filter_size, dropout_rate, add_dropout_fc_layer):
        super(MyNetwork, self).__init__()
        # Define layers and operations in the network
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size[0])
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size[1])
        self.conv2_drop = nn.Dropout2d(p=dropout_rate) if add_dropout_fc_layer else nn.Identity()

        # Calculate the flattened tensor size
        x = torch.randn(1, 1, 28, 28)  # Dummy input tensor
        x = F.relu(F.max_pool2d(self.conv1(x), pool_filter_size[0]))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), pool_filter_size[1]))
        flattened_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(flattened_size, 50)
        self.fc2 = nn.Linear(50, 10)
        self.add_dropout_fc_layer = add_dropout_fc_layer

    def forward(self, x, pool_filter_size):
        """Computes a forward pass for the network."""
        x = F.relu(F.max_pool2d(self.conv1(x), pool_filter_size[0]))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), pool_filter_size[1]))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        if self.add_dropout_fc_layer:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch, network, optimizer, train_loader, pool_filter_size, device, log_interval=10):
    """Train the neural network for one epoch."""
    network.train()
    train_losses = []
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data to GPU
        optimizer.zero_grad()
        output = network(data, pool_filter_size)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())

    train_accuracy = 100. * train_correct / train_total
    return train_losses, train_accuracy

def test(network, test_loader, pool_filter_size, device):
    """Evaluate the performance of the neural network on the test dataset."""
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data to GPU
            output = network(data, pool_filter_size)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy

# Main function
def main():
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if CUDA is available
    print(f"Using device: {device}")

    # Define parameter ranges
    conv_filter_sizes_range = [(3, 3), (5, 5), (7, 7)]
    pool_filter_sizes_range = [(2, 2)]
    epochs_range = [5, 6, 7]
    dropout_rates_range = [0.1, 0.3, 0.5, 0.6]
    add_dropout_fc_layer_range = [True, False]
    batch_size_train_range = [32, 64]

    n_variations = 0
    for conv_filter_size in conv_filter_sizes_range:
        for pool_filter_size in pool_filter_sizes_range:
            for epochs in epochs_range:
                for add_dropout_fc_layer in add_dropout_fc_layer_range:
                    for batch_size_train in batch_size_train_range:
                        if add_dropout_fc_layer:
                            for dropout_rate in dropout_rates_range:
                                n_variations += 1
                        else:
                            n_variations += 1

    print(f"Total number of variations: {n_variations}")

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    batch_size_test = 1000  # Set the batch size for testing
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # Data loading and preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    top_5_combinations = []
    test_results = {}
    
    experiment_number = 0  # Counter for the experiment number

    for conv_filter_size in conv_filter_sizes_range:
        for pool_filter_size in pool_filter_sizes_range:
            for epochs in epochs_range:
                for add_dropout_fc_layer in add_dropout_fc_layer_range:
                    for batch_size_train in batch_size_train_range:
                        if add_dropout_fc_layer:
                            for dropout_rate in dropout_rates_range:
                                experiment_number += 1  # Increment the experiment number
                                print(f"\nExperiment {experiment_number}/{n_variations}")
                                print("\n========================================================")
                                print(f"Hyperparameters: Conv Filter Size: {conv_filter_size}, Pool Filter Size: {pool_filter_size}, Epochs: {epochs}, Dropout Rate: {dropout_rate}, Add Dropout FC Layer: {add_dropout_fc_layer}, Batch Size Train: {batch_size_train}")
                                print("========================================================\n")
                                
                                train_loader = torch.utils.data.DataLoader(
                                    torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
                                    batch_size=batch_size_train, shuffle=True)

                                test_loader = torch.utils.data.DataLoader(
                                    torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform),
                                    batch_size=batch_size_test, shuffle=False)

                                network = MyNetwork(conv_filter_size, pool_filter_size, dropout_rate, add_dropout_fc_layer).to(device)
                                optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

                                test_loss = 0
                                train_losses = []
                                train_accuracy = 0
                                test_accuracy = 0

                                start_time = time.time()  # Start timing

                                for epoch in range(1, epochs + 1):
                                    epoch_train_losses, epoch_train_accuracy = train(epoch, network, optimizer, train_loader, pool_filter_size, device, log_interval)
                                    train_losses.extend(epoch_train_losses)
                                    train_accuracy = epoch_train_accuracy
                                    test_loss, test_accuracy = test(network, test_loader, pool_filter_size, device)

                                end_time = time.time()  # End timing
                                duration = end_time - start_time  # Calculate duration

                                test_results[(conv_filter_size, pool_filter_size, epochs, dropout_rate, add_dropout_fc_layer, batch_size_train)] = (train_accuracy, test_accuracy, duration)
                        else:
                            dropout_rate = None  # No dropout rate if there's no dropout layer
                            experiment_number += 1  # Increment the experiment number
                            print(f"\nExperiment {experiment_number}/{n_variations}")
                            print("\n========================================================")
                            print(f"Hyperparameters: Conv Filter Size: {conv_filter_size}, Pool Filter Size: {pool_filter_size}, Epochs: {epochs}, Dropout Rate: {dropout_rate}, Add Dropout FC Layer: {add_dropout_fc_layer}, Batch Size Train: {batch_size_train}")
                            print("========================================================\n")
                            
                            train_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform),
                                batch_size=batch_size_train, shuffle=True)

                            test_loader = torch.utils.data.DataLoader(
                                torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform),
                                batch_size=batch_size_test, shuffle=False)

                            network = MyNetwork(conv_filter_size, pool_filter_size, dropout_rate, add_dropout_fc_layer).to(device)
                            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

                            test_loss = 0
                            train_losses = []
                            train_accuracy = 0
                            test_accuracy = 0

                            start_time = time.time()  # Start timing

                            for epoch in range(1, epochs + 1):
                                epoch_train_losses, epoch_train_accuracy = train(epoch, network, optimizer, train_loader, pool_filter_size, device, log_interval)
                                train_losses.extend(epoch_train_losses)
                                train_accuracy = epoch_train_accuracy
                                test_loss, test_accuracy = test(network, test_loader, pool_filter_size, device)

                            end_time = time.time()  # End timing
                            duration = end_time - start_time  # Calculate duration

                            test_results[(conv_filter_size, pool_filter_size, epochs, dropout_rate, add_dropout_fc_layer, batch_size_train)] = (train_accuracy, test_accuracy, duration)

    # Print out train and test accuracies for each combination
    for params, (train_acc, test_acc, duration) in test_results.items():
        print(f'Parameters: {params}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, Training Time: {duration:.2f} seconds')

    # Get the top 5 combinations with better test accuracy
    sorted_results = sorted(test_results.items(), key=lambda x: x[1][1], reverse=True)[:5]
    print("\nTop 5 combinations with better test accuracy:")
    for rank, (params, (train_acc, test_acc, duration)) in enumerate(sorted_results, start=1):
        print(f"Rank {rank}: Parameters: {params}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%, Training Time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
