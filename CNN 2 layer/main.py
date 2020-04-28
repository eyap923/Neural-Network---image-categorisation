import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.conv import FashionCNN2
from models.conv3layer import FashionCNN3
from models.conv4layer import FashionCNN4
from models.VGG import VGG

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
from PIL import Image

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


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
    #### Data Pre-processing + Loading Training & Test Data ###

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


    """Change Model here"""
    model = FashionCNN4()
    model_name = FashionCNN4
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Models used
    #global batch_size, num_epochs, learning_rate, optimizer, scheduler
    if (model_name == FashionCNN2):
        batch_size = 100
        num_epochs = 5
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif (model_name == FashionCNN3):
        batch_size = 100
        num_epochs = 30
        learning_rate = 0.015
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    elif (model_name == FashionCNN4):
        batch_size = 256
        num_epochs = 10
        learning_rate = 0.001
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    elif (model_name == VGG):
        batch_size = 512
        num_epochs = 20
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


    # Using data predefined loader
    # Combines a dataset and a sampler, and provides an iterable over the given dataset.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    #num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    error = nn.CrossEntropyLoss()

    #learning_rate = 0.001
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(model)

    n_iters = 5500
    num_epochs = n_iters / (len(train_data) / batch_size)
    num_epochs = int(num_epochs)
    count = 0;
    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            print(count)
            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            count += 1
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                # Print Loss
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(iter, loss.data, accuracy))

        if ((model == FashionCNN3) or (model == VGG)):
            scheduler.step()






    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")




if __name__ == '__main__':
    main()
