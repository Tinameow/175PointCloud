from model_utils import *
from multiViewNet import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def check_accuracy(model, loader, train = False):
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
    if train:
        print("Training : ", end='')
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def train(model, loss_fn, optimizer, train_loader, val_loader, num_epochs=1, save=False, filename="save_"):
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
            torch.save(model.state_dict(), filename + str(epoch) + ".pth")



if __name__ == '__main__':
    cams = get_cams("4cams")

    train_transforms = transforms.Compose([
        PointSampler(512),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        create_data_point(cams),
        ToTensor()
    ])


    default_transforms = transforms.Compose([
            PointSampler(512),
            Normalize(),
            create_data_point(cams),
            ToTensor()
        ])

    train_ds = PointCloudData(Path(path), transform=train_transforms)
    valid_ds = PointCloudData(Path(path), valid=True, folder='test', transform=default_transforms)



    pointnet = MVNet()
    pointnet.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pointnet.parameters(), lr=0.1, momentum=0.9)

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

    train(pointnet, loss_fn, optimizer, train_loader, valid_loader, num_epochs=50, save=True, filename="cam4_512_")