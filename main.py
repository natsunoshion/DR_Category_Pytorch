import torch
import numpy as np
import torchvision
import adabound
from tqdm import tqdm  # 进度条
from torch import nn
from torch.utils.data import DataLoader  # 创建可以迭代的数据装载器
from torchvision import datasets  # 调用数据集
from torchvision import transforms  # 图像预处理包
import torch.nn.functional as F  # 调用激活函数ReLU
import torch.optim as optim  # 使用优化器
import matplotlib.pyplot as plt
import torchvision.models as models  # 预训练模型
from FocalLoss import FocalLoss  # 损失函数
from ResNet_CAB import resnet50

##################################################
# 本次 python 大作业是 DR 图像分类
# 基础模型为 ResNet50
# 我在上面加了 CAB 注意力机制和 CBAM 注意力机制
##################################################


# 寻找最优算法
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


##################################################
# 混淆矩阵的 kappa 参数计算
##################################################
def weight_kappa(result, test_num):
    weight = torch.zeros(5, 5)  # 新建一个矩阵，存放权重
    for i in range(5):
        for j in range(5):
            weight[i, j] = (i - j) * (i - j) / 16
    fenzi = 0
    for i in range(5):
        for j in range(5):
            fenzi = fenzi + result[i, j] * weight[i, j]
    fenmu = 0
    for i in range(5):
        for j in range(5):
            fenmu = fenmu + weight[i, j] * result[:, j].sum() * result[i, :].sum()

    weght_kappa = 1 - (fenzi / (fenmu / test_num))
    return float(weght_kappa)



##################################################
# 混淆矩阵的绘制
##################################################
def plot(matrix, acc):
    classes = 5  # 5 分类
    labels = ['No DR', 'Mild DR', 'Moderate DR', 'Servere DR', 'Proliferative DR']  # 标签

    plt.imshow(matrix, cmap=plt.cm.Blues)

    # 设置x轴坐标label
    plt.xticks(range(classes), labels, rotation=45)
    # 设置y轴坐标label
    plt.yticks(range(classes), labels)
    # 显示colorbar
    plt.colorbar()
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion matrix (acc=' + str(acc) + ')')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(classes):
        for y in range(classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            plt.text(x, y, info,
                        verticalalignment='center',
                        horizontalalignment='center',
                        color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.show()
    plt.savefig('result/confusion_matrix.png')
    plt.clf()  # 清空画布，免得下一次绘图的时候出问题


##################################################
# 训练
##################################################
def train(epoch):
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    running_loss = 0.0
    right = 0
    for batch_idx, data in loop:
        # 初始化
        inputs, target = data
        inputs =inputs.cuda()
        target = target.cuda()
        optimizer.zero_grad()  # 优化器使用之前先清零

        # 优化模型
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # 计算评估值，并用进度条、混淆矩阵的方式输出
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        # 累加识别正确的样本数
        right += (predicted == target).sum()
        # 更新信息
        loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
        loop.set_postfix(loss=running_loss / (batch_idx + 1), acc=float(right) / float(batch_size * (batch_idx + 1)))
    return loss.item()


##################################################
# 测试
##################################################
def test():
    model.eval()
    correct = 0
    total = 0

    # 混淆矩阵
    conf_matrix = torch.zeros(5, 5)

    with torch.no_grad():  # 全程不计算梯度
        for data in test_loader:
            images, labels = data  # 记录原始值
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)  # 把图像丢进模型训练，得到输出结果
            _, predicted = torch.max(outputs.data, dim=1)  # 求每一行最大值的下标
            # 输出的 outputs 是 N * 1 的，dim=1 指的是沿着第 1 个维度，-->，即列，行是第 0 个维度
            # 返回两个值，最大值下标以及最大值是多少
            total += labels.size(0)  # 记录有多少条 labels
            correct += (predicted == labels).sum().item()  # 相同的求和取标量

            # 混淆矩阵计算，行是预测值，列是真实值，对应的矩阵值是这样的有序对在测试集中出现的次数
            for i in range(labels.size()[0]):
                conf_matrix[int(outputs.argmax(1)[i])][int(labels[i])] += 1
            # print('本次计算后混淆矩阵为：\n', conf_matrix.int())
    conf_matrix = conf_matrix.int()  # 转为 int 矩阵，方便在终端看
    print('Accuracy on test set: %d %% ' % (100 * correct / total))
    print('混淆矩阵为：\n', conf_matrix)
    print('weight_kappa的值为：', weight_kappa(conf_matrix, len(test_data)))

    plot(conf_matrix, correct / total)  # 绘制

    return correct / total


##################################################
# 主函数
##################################################
if __name__ == '__main__':
    image_size = 512  # 图像大小 512 * 512
    batch_size = 8  # 调这么低是因为云服务器内存有限，调太高直接给我 CUDA out of memory
    EPOCHS = 70  # 测 70 次
    lr1 = 1e-3
    lr2 = 0.1

    ######################
    # 预处理
    ######################
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 使用 torchvision.datasets.ImageFolder 读取数据集指定 train 和 test 文件夹
    train_data = torchvision.datasets.ImageFolder('DDR/train/', transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)  # Windows 下 num_workers 只能设置为 0，当然，我用云服务器就没这个限制了

    test_data = torchvision.datasets.ImageFolder('DDR/test/', transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


    ######################
    # 实例化
    ######################
    # 新版 ResNet 对输入图像大小没强制要求了，因为有一个 model.avgpool = nn.AdaptiveAvgPool2d((1, 1))，这也就是最后不管卷积层的输出大小，直接取平均值进行输出
    # model = ResidualNet("ImageNet", 34, 5, 'BAM')  # 带有 BAM 注意力机制的 ResNet
    model = resnet50()  # 带有 CBAM 注意力机制的 ResNet
    # model = ghost_net()  # GhostNet 自带 SE 注意力机制，dropout 层
    # model = mobilenetv3()  # 自带 drpout、SE
    # 下面这两行是在重写网络的全连接层，把输出调为 5，也就是做 5 分类工作
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 5)

    # 加一个 dropout 层
    model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))

    # gpu 加速
    model = model.cuda()

    ######################
    # 损失函数
    ######################
    # 带权损失函数的方案已经弃用
    # class_weight = torch.tensor([1880 / 1880, 1880 / 189, 1880 / 1344, 1880 / 71, 1880 / 275])  # 损失函数的权值，由样本数量构成，好像效果不太行
    # criterion = nn.CrossEntropyLoss(weight=class_weight)  # 使用带权值的交叉熵函数，解决样本数量不平衡的问题
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(5)  # 使用 FocalLoss 函数来处理不平衡数据，实际测试效果最好
    criterion = criterion.cuda()

    ######################
    # 优化器
    ######################
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)  # momentum:使用带冲量的模型来优化训练过程
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)  # 加了 L2 正则化，试试效果，好像不咋样
    optimizer = adabound.AdaBound(model.parameters(), lr=lr1, final_lr=lr2)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (epoch + 1), last_epoch=-1)  # 自动调整学习率

    ######################
    # 训练部分
    ######################
    torch.cuda.empty_cache()  # 清掉缓存，云服务器老是崩，不过这句话好像没什么用，还不如调小 batch_size
    loss_list = []
    acc_list = []
    epoch_list = []

    for epoch in range(EPOCHS):
        # 由于使用了进度条，这个也被废用了
        # print("----------第{}轮训练开始：----------".format(epoch + 1))
        loss = train(epoch)  # 训练一次
        accuracy = test()  # 测试一次

        loss_list.append(loss)
        acc_list.append(accuracy)
        epoch_list.append(epoch)

    ######################
    # 绘图
    ######################
    plt.plot(epoch_list, loss_list)
    plt.plot(epoch_list, acc_list)
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('result/loss-acc.png')
    plt.clf()