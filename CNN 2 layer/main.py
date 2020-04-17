import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.conv import FashionCNN2


import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
#from torch.optim.lr_scheduler import StepLR
from PIL import Image


# Use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")


def main():
    save_model = True
    #####################    Data Pre-processing + Loading Training & Test Data

    # Data Augmentation
    # train_transform = transforms.Compose([transforms.RandomResizedCrop(28),transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomVerticalFlip(), transforms.RandomPerspective(),
    #                                       transforms.ToTensor()])

    # No data augmentation is being performed on the test data - we want this to keep its representation of the classes
    # test_transform = transforms.Compose(transforms.ToTensor())

    # For CUDA
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Choosing FashionMNIST dataset
    train_data = datasets.FashionMNIST('data/training', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST('data/test', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Using data predefined loader
    # Combines a dataset and a sampler, and provides an iterable over the given dataset.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, **kwargs)

    num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    model = FashionCNN2()
    model.to(device)

    error = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            print(count)
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

            # Testing the model
            if not (count % 50):  # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 500):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))



    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")




if __name__ == '__main__':
    main()
