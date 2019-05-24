import argparse
import torch
import torchvision.transforms as transforms
import resnet
from dataloader import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Net')
# !! Must provide when running main
parser.add_argument('--data_dir', default='../data/ISIC2019/')

# FOR TRAIN
parser.add_argument('--resize_img', default=300, type=int)
parser.add_argument('--model',default='resnet50')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--n_epochs', type=int, default=250)
parser.add_argument('--nclass', type=int, default=9)
parser.add_argument('--GPU_ids', default=0)


if __name__ == '__main__':
    # Dataset
    img_size = 224
    resize_img = 300
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(resize_img),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1], scale=[0.7, 1.3]),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    print('==> Preparing data..')
    trainset = dataloader(train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=50, shuffle=True)

    model = args.model
    # Use args.model as pretrain model
    if model == 'resnet152':
        net = resnet.resnet152().to(device)
    elif model == 'resnet50':
        net = resnet.resnet50().to(device)
    else:
        sys.exit(-1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(start_epoch, n_epochs):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
