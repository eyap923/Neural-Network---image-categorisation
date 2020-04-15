import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix



# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(); optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def test(model, device, test_loader):

    model.eval()

    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.show()

    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.show()

    # Accuracy check
    total_correct = [0. for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = Variable(images)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()

            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1

    for i in range(10):
        print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))

    from itertools import chain

    predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
    labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
    predictions_l = list(chain.from_iterable(predictions_l))
    labels_l = list(chain.from_iterable(labels_l))
    import sklearn.metrics as metrics

    confusion_matrix(labels_l, predictions_l)
    print("Classification report for CNN :\n%s\n"
          % (metrics.classification_report(labels_l, predictions_l)))


def main():
    num_epochs = 5
    count = 0
    # Lists for visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    # #####################    This is where data procressing and stuff goes







    # #####################    Build your network and run   ############################
    model = FashionCNN()
    model.to(device)
    error = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
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
