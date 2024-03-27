import torch
import torchvision
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

# Load the pre-trained model
model = Net()
model.load_state_dict(torch.load('./results/model.pth'))
model.eval()  # Set model to evaluation mode

# Load the test dataset
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=1, shuffle=False)

# Function to plot the first 9 digits with predictions
def plot_predictions(images, predictions):
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')  # Remove the extra dimension
        ax.set_title(f'Prediction: {predictions[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Iterate through the first 10 examples in the test set
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        if i >= 10:
            break
        output = model(image)
        probabilities = torch.exp(output).squeeze().tolist()
        predicted_label = torch.argmax(output, dim=1).item()
        correct_label = label.item()
        print(f"Example {i+1}:")
        print("Probabilities:", [f"{prob:.2f}" for prob in probabilities])
        print("Predicted Label:", predicted_label)
        print("Correct Label:", correct_label)
        print()

# Plot the first 9 digits with predictions
images = [image.squeeze().numpy() for image, _ in test_loader][:9]
images = [torch.tensor(image).unsqueeze(0) for image in images]  # Convert to tensor and add batch dimension
predictions = [torch.argmax(model(image), dim=1).item() for image in images]
plot_predictions(images, predictions)
