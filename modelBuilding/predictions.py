import torch
from torchaudioIntro import FeedForwardNet, download_mnist_datasets

class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def predict(model, input, target, class_mapping):
    model.eval() # turns on evaluation mode for the model, meaning that it doesn't train or adjust the weights in the nn
    
    # Here, we need to use a context manager
    with torch.no_grad(): # This tells the model to not calculate gradients, since we are not training
        predictions = model(input)
        # Tensor(number of samples being passed to the model (batch size), number of classes that are being predicted) 
        # so the size of the tensor is batch size * number of classes
        predicted_index = predictions[0].argmax(0)
        # map predicted index to relative class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
        
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("modelBuilding/feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)
    # class_mapping performs a maping from the integer returned by the nn to the desired class

    print(f"Predicted: '{predicted}', expected: '{expected}'")