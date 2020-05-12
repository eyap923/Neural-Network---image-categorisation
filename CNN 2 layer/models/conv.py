import torch.nn as nn

# The network should inherit from the nn.Module
##This is the 2 Layer CNN

# batch normalization to preprocess every layer of the network by adjusting snd scaling the activations

# Conv2D layers are used for the convolution operation that extracts features from the input images by sliding a convolution filter over the input to produce a feature map.
# Maxpooling used to reduce the dimensionality of each feature, which helps shorten training time and reduce number of parameters.
class FashionCNN2(nn.Module):

    def __init__(self):
        super(FashionCNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

