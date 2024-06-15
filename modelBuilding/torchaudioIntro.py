import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
The point of this file is to provide a basic layout of the process of developing a neural network. With everything that might 
confuse me commented and explained, so that I can revisit this file whenever I need to refresh myself with the basics of deep learning.
That being said ...
Here are the 5 steps to building a NN model with pytorch
1. Download Dataset
2. Create data loader
3. Build model
4. Train
5. Save Trained model
"""

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .005

class FeedForwardNet(nn.Module): # Creates a FeedForwardNet neural network. Inherits from Module base class

    def __init__(self):
        super().__init__() # calls constructor parent class
        self.flatten = nn.Flatten() # "Flattens" an input data (image) into a 1D vector
        self.dense_layers = nn.Sequential( # Sequential container of layers
            nn.Linear(28*28, 256), # Defines an input layer of 28 * 28 nodes connected to one of 256 nodes
            nn.ReLU(), # Wraps a ReLU (Max(0, num)) activation function to the first hidden layer
            nn.Linear(256, 10) # Connext the hidden layer to the outputs layer (logits)
        )
        self.softmax = nn.Softmax(dim=1) # converts the logits to probabilities

    def forward(self, input_data): # This function specifies how the input data should be processed through the layers
        flattened_data = self.flatten(input_data) 
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions

def download_mnist_datasets():
    train_data = datasets.MNIST( #MNIST is a dataset that comes from the library
        root="data", # Stores data in the "data" directory
        download=True, # If the dataset has not been downloaded yet, it will need it to be downloaded
        train=True, # Tells pytorch that we are interested in the training part of the dataset
        transform=ToTensor() # Allows us to do transformations to the dataset
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data

def train_one_epoch(model, data_loader, loss_fn, optimiser, device): # remember and epoch is a passthrough all the dataset
    for inputs, targets in data_loader: # targets are the expected values from the respective input
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad() # resets the gradient at each new iteration
        loss.backward() # Backpropagation
        optimiser.step() # updates the weights

    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------")
    print("Training is done")
    

if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device="mps"
    else:
        device = "cpu"
    print(f"Using {device} device")
    feed_forward_net = FeedForwardNet().to(device)

    # instantiate loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(feed_forward_net, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored at feedforwardnet.pth")