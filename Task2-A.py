import torch
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

# Print the model structure
print(model)

# Analyze the first layer
first_layer_weights = model.conv1.weight.data
print("Shape of the first layer weights:", first_layer_weights.shape)

# Visualize the ten filters
fig = plt.figure(figsize=(10, 8))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.imshow(first_layer_weights[i, 0].cpu(), cmap='plasma')
    plt.title(f'Filter {i}')
    plt.xticks([])
    plt.yticks([])
plt.show()
