from PointNet import *
from pointNet_utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from path import Path

def train(model, train_loader, val_loader=None,  epochs=1, save=True):
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=3)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: (%.2f)' % val_acc)
            scheduler.step(val_acc)

        # save the model
        if save:
            torch.save(model.state_dict(), "save_3_"+str(epoch)+".pth")

def check_accuracy(model, loader, train = False):
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for i, data in enumerate(loader):
        x_var, y = data['pointcloud'].to(device).float(), data['category']

        scores, __, __ = model(x_var.transpose(1,2))
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    if train:
        print("Training : ", end='')
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


if __name__ == "__main__":

    train_transforms = transforms.Compose([
        PointSampler(1024),
        Normalize(),
        RandRotation_z(),
        RandomNoise(),
        ToTensor()
    ])
    train_ds = PointCloudData(path, transform=train_transforms)
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=default_transforms)

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    pointnet = PointNet()
    pointnet.load_state_dict(torch.load('save_3_19.pth'))
    pointnet.to(device)
    check_accuracy(pointnet, valid_loader)

    # optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(pointnet.parameters(), lr=0.001, momentum=0.9)

    # train(pointnet, train_loader, valid_loader, epochs=30, save=True)