import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""
1. Download Dataset
2. Create data loader
3. Build model
4. Train
5. Save Trained model
"""

BATCH_SIZE = 128

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.SoftMax(dim=1)

    def forward(self, input_data):
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

if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset downloaded")

    # create a data loader for the train set
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)