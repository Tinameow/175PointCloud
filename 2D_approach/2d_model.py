from model_utils import *


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for i, data in enumerate(loader):
        x_var, y = data['pointcloud'].to(device).float(), data['category']

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def train(model, loss_fn, optimizer, train_loader, val_loader, num_epochs=1, save=False):
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=3)

    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, data in enumerate(train_loader):
            x_var, y_var = data['pointcloud'].to(device).float(), data['category'].to(device)

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % 10 == 0:
                print('loss = %.4f' % (loss.data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = check_accuracy(model, val_loader)
        scheduler.step(val_acc)

        if save:
            torch.save(model.state_dict(), "save_" + str(epoch) + ".pth")


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
        layer1 = self.conv1(input[:, 0, :, :][:, None, :, :])
        layer2 = self.conv1(input[:, 1, :, :][:, None, :, :])
        layer3 = self.conv1(input[:, 2, :, :][:, None, :, :])

        xb = nn.MaxPool1d(3)(torch.stack((layer1, layer2, layer3), 2))
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

if __name__ == '__main__':
    cam1 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 2, 0]]).T, R=makerotation(90, 0, 0))
    cam2 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[2, 0, 0]]).T, R=makerotation(0, 90, 0))
    cam3 = Camera(f=25, c=np.array([[16, 16]]).T, t=np.array([[0, 0, 2]]).T, R=makerotation(180, 0, 0))
    cams = [cam1, cam2, cam3]

    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        create_data_point(cams),
        ToTensor()
    ])


    default_transforms = transforms.Compose([
            PointSampler(1024),
            Normalize(),
            create_data_point(cams),
            ToTensor()
        ])

    train_ds = PointCloudData(Path(path), transform=train_transforms)
    valid_ds = PointCloudData(Path(path), valid=True, folder='test', transform=default_transforms)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    pointnet = MVNet()
    pointnet.load_state_dict(torch.load('save_22.pth'))
    pointnet.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pointnet.parameters(), lr=0.001, momentum=0.9)

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

    train(pointnet, loss_fn, optimizer, train_loader, valid_loader, num_epochs=50, save=False)