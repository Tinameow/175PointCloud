import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvOneView(nn.Module):
    def __init__(self):
        '''extracting features from single view'''
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(2304, 1024)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        # input.shape == (bs,1,32,32)

        # bs = input.size(0)
        xb = self.pool(self.relu(self.bn1(self.conv1(input))))
        xb = self.pool(self.relu(self.bn2(self.conv2(xb))))

        flat = nn.Flatten()(xb)
        xb = self.relu(self.bn3(self.fc1(flat)))

        return xb


class CombineMultiView(nn.Module):
    '''extracting features from multi views'''

    def __init__(self):
        super().__init__()
        self.conv1 = ConvOneView()

    def forward(self, input):
        # print(list(input[:,0,:,:][:,None,:,:].size()))
        n = list(input.size())[1]
        layers = []
        for i in range(n):
            layer = self.conv1(input[:, i, :, :][:, None, :, :])
            layers.append(layer)

        xb = nn.MaxPool1d(n)(torch.stack(layers, 2))
        # xb = torch.stack(layers, 1)
        # xb = nn.Conv1d(n, 1, 1)(xb)
        # print(list(xb.size()))

        output = nn.Flatten(1)(xb)

        return output


class MVNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.CombineMultiView = CombineMultiView()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb = self.CombineMultiView(input)
        xb = nn.ReLU(inplace=True)(self.bn1(self.fc1(xb)))
        xb = nn.ReLU(inplace=True)(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output)


