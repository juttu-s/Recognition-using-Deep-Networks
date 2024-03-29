
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    
from zipfile import ZipFile
file_name="greek_train.zip"
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Done')

# Load the pre-trained MNIST network
mnist_model = Net()
mnist_model.load_state_dict(torch.load('model.pth'))

# Freeze the parameters for the whole network
for param in mnist_model.parameters():
    param.requires_grad = False

# Replace the last layer with a new Linear layer with three nodes
mnist_model.fc2 = nn.Linear(50, 3)  # the second last layer has 50 nodes

# greek data set transform to transform the RGB images to grayscale, scale and crop them to the correct size,
# and invert the intensities to match the MNIST digits.
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# DataLoader for the Greek letter dataset
training_set_path="/content/greek_train"

greek_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(training_set_path,
                                      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                GreekTransform(),
                                                                                torchvision.transforms.Normalize(
                                                                                    (0.1307,), (0.3081,))])),
    batch_size=5,
    shuffle=True)

# Define optimizer and loss function
optimizer = optim.SGD(mnist_model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()
num_epochs = 5

# Lists to store training loss
train_losses = []

# Train the modified network on the Greek letter dataset
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in greek_train:
        optimizer.zero_grad()
        outputs = mnist_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(greek_train)
    print(f"Epoch {epoch + 1}, Loss: {train_loss}")
    train_losses.append(train_loss)

# Plot training loss
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Printout of the modified network
print(mnist_model)  # Print model architecture after modifications