import torch
from torch.nn import Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from dash import html
from torch import nn


class ImageNN(Module):

    def __init__(self, n_inputs=784):
        super(ImageNN, self).__init__()

        self.linear1 = nn.Linear(n_inputs, 512)
        kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()

        self.linear2 = nn.Linear(512, 256)
        kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()

        self.linear3 = nn.Linear(256, 256)
        kaiming_uniform_(self.linear3.weight, nonlinearity='relu')
        self.act3 = nn.ReLU()

        self.linear4 = nn.Linear(256, 128)
        xavier_uniform_(self.linear4.weight)
        self.act4 = nn.ReLU()

        self.linear5 = nn.Linear(128, 10)
        xavier_uniform_(self.linear5.weight)

    def forward(self, X):
        X = torch.flatten(X, 1)

        output = self.linear1(X)
        output = self.act1(output)

        output = self.linear2(output)
        output = self.act2(output)

        output = self.linear3(output)
        output = self.act3(output)

        output = self.linear4(output)
        output = self.act4(output)

        output = self.linear5(output)
        output = nn.functional.softmax(output, dim=1)

        return output


def train_nn(device, model, train_loader, optimizer, loss_function, epoch):
    for batch, (X, y) in enumerate(train_loader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # predict
        prediction = model(X)

        # loss
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch * len(X), len(train_loader.dataset),
                       100. * batch / len(train_loader), loss.item()))


def test_nn(device, model, test_loader, loss_function):
    model.eval()  # Set the model in evaluation mode
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            output = model(X)

            loss = loss_function(output, y)

            test_loss += loss.item()

            _, predicted = torch.max(output.data, 1)

            correct += (predicted == y).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%\nTest loss: {test_loss}\n")


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


def open_image(filename):
    try:
        image = np.array(Image.open(filename))
    except UnidentifiedImageError:
        return html.Div(
            [
                "Please load a valid image file!!"
            ]
        )

    return image


device = ('cuda' if torch.cuda.is_available() else 'cpu')


def predicted_digit(model, image):
    # Transform the image
    image = transform(image).unsqueeze(0).to(device)

    # Perform the prediction
    with torch.no_grad():
        model.eval()
        output = model(image)

    # Get the predicted digit
    _, predicted = torch.max(output.data, 1)
    digit = predicted.item()

    return digit
