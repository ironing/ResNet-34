import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from resnet import resnet34

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# 输出保存路径
parser.add_argument('--outf', default='.models/', help='folder to output images and model checkpoints')
# 恢复训练时的模型路径
parser.add_argument('--net', default='.models/resnet34.pth', help="path to net (to continue training")
args = parser.parse_args()

# 超参数设置
EPOCH = 135
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.1

# 准备数据集并预处理
transforms_train = transforms.Compose([
    # 四周填充0，再随机剪成32 * 32
    transforms.RandomCrop(32,padding=4),
    # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # R,G,B每层的归一化用到的均值和方差
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transforms=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transforms=transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar10的label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义
net = resnet34().to(device)

# 损失函数和优化方式定义
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    best_acc = 85
    print("Start Training, Resnet-34")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(pre_epoch,EPOCH):
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

                    # 每训练一个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: % .03f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i+1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f |Acc: %.03f'
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i+1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Wainting Test")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        # 取得分最高的那个类
                        _,predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为: %.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    # 将结果写入acc.txt文件中
                    print("Saving model......")
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH = %03d,Accuracy = %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH = %d,best_acc = .03f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, Total epoch = %d" % EPOCH)








