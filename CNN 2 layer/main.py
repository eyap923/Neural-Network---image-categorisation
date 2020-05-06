import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.conv import FashionCNN2
from models.conv3layer import FashionCNN3
from models.conv4layer import FashionCNN4
from models.VGG import VGG

from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from itertools import chain

# Use GPU if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def main():
    save_model = True

    """----------------------Change Model here----------------------"""
    model = FashionCNN2()
    model_name = FashionCNN2
    """-------------------------------------------------------------"""
    model.to(device)

    # Models used
    # global batch_size, num_epochs, learning_rate, optimizer, scheduler
    # All use learning rate decay to improve performance
    # Batch size chosen ro not use too much memory
    # Number of iterations chosen when accuracy begins to flatten
    # optimizer is responsible for updating the weights of the neurons via backpropagation.
    # It calculates the derivative of the loss function with respect to each weight and subtracts it from the weight.

    if (model_name == FashionCNN2):
        batch_size = 100
        n_iters = 2600
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    elif (model_name == FashionCNN3):
        batch_size = 100
        n_iters = 2600
        learning_rate = 0.015
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    elif (model_name == FashionCNN4):
        batch_size = 256
        n_iters = 600
        learning_rate = 0.001
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    elif (model_name == VGG):
        batch_size = 512
        n_iters = 300
        learning_rate = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)



    # For CUDA
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Choosing FashionMNIST dataset
    train_data = datasets.FashionMNIST('data/training', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.FashionMNIST('data/test', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))


    # loss function
    # cross-entropy loss calculates the error rate between the predicted value and the original value
    criterion = nn.CrossEntropyLoss()

    # Using data predefined loader
    # Combines a dataset and a sampler, and provides an iterable over the given dataset.
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    # For visualization of loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # For computing confusion matrix, precision, recall and f1-score
    predictions_list = []
    labels_list = []

    print(model)

    # Basing epochs off iterations & batch_size to avoid using too much memory
    num_epochs = n_iters / (len(train_data) / batch_size)
    num_epochs = int(num_epochs)
    count = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            print(count)

            # Converts to Tensor
            # https://pytorch.org/docs/stable/autograd.html
            # https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable
            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            # Clear previously calculated gradients (if any)
            # http://seba1511.net/tutorials/beginner/blitz/neural_networks_tutorial.html
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)
            count += 1

            # Calculate Loss: softmax --> cross entropy loss
            # To find out how far is the result from ground truth/target
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            # Back propagation - compute gradient of each parameter
            # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350
            loss.backward()

            # Updating parameters based on gradient calculated
            optimizer.step()

            # Printing every 50 iterations to monitor results
            if not (count % 50):
                # Calculate Accuracy
                correct = 0
                total = 0

                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)
                    labels_list.append(labels)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)
                    predictions_list.append(predicted)

                    # Total number of labels
                    total += labels.size(0)

                    correct += (predicted == labels).sum()
                    accuracy = 100 * correct / total


                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

                # Print loss and accuracy
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))


        if ((model == FashionCNN3) or (model == VGG)):
            scheduler.step()

    loss_list.append(loss.data)
    iteration_list.append(count)
    accuracy_list.append(accuracy)
    
    # Print loss and accuracy at the end once again (if the end count isn't % 50 == 0)
    print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

    # Converts each Tensor values in the list to a list of integer values (lists within a list)
    predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
    labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]

    # Converts the list of lists to a single big list
    predictions_l = list(chain.from_iterable(predictions_l))
    labels_l = list(chain.from_iterable(labels_l))

    # Confusion Matrix, precision, recall & f1-score
    print(metrics.confusion_matrix(labels_l, predictions_l))
    print("Classification report for CNN :\n%s\n" % (metrics.classification_report(labels_l, predictions_l)))

    # To illustrate the learning process
    plt.figure(1)
    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.show()

    plt.figure(2)
    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterations vs Accuracy")
    plt.show()

    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")




if __name__ == '__main__':
    main()
