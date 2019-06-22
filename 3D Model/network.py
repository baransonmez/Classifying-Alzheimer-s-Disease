import torch.nn as nn
import torch.nn.functional as F


class CustomNet(nn.Module):
    def __init__(self, n_classes=2):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(176, 8, kernel_size=(3, 3), stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.fc1 = nn.Linear(488072, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # result consists 488072 neurons
        x = x.view(-1, 488072)
        x = F.relu(F.dropout(self.fc1(x)))
        x = self.fc2(x)

        return x
