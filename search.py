"""
    author: He Jiaxin
    date: 14/8/2019
    version: 1.0
    function: search the best PANet architecture.
"""
import torch
import torch.nn as nn
from controller import StateSpace, RNN, Controller
from path_aggregation import PathAggregation
import resnet
import torchvision
import torchvision.transforms as transforms
from optimizer import NAS_Adam
import numpy as np
from torch.optim import Adam


"""
    在分类任务cifar10上初步测试
"""

# 超参定义区
PATH = '/Users/hejiaxin/DeepLearning/data'
MAX_TRAIL = 250
EXPLORATION = 0.8
INPUT_SIZE = 20
HIDDEN_SIZE = 32
NUM_LAYERS = 2
LEARNING_RATE = 0.01
BETAS = (0.5, 0.99)
WEIGHT_DECAY = 1e-3
PANET_NUM_EPOCH = 5000
ACC_BETA = 0.8


# 得到cifar10数据集
transform = transforms.Compose([transforms.Pad(4),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
                                transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root=PATH, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root=PATH, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
total_step = len(train_loader)


# 初始化RNN controller和StateSpace
state_space = StateSpace()
rnn = RNN(state_space, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
controller = Controller(state_space, rnn, exploration=EXPLORATION)
state = np.random.randint(0, 4)
beta = ACC_BETA
beta_bias = ACC_BETA
moving_acc = 0.0
resnet50 = resnet.ResNet50()
resnet101 = resnet.ResNet101()


# 初始化optimizer
linear_param = []
for linear in rnn.linears:
    for param in linear.parameters():
        linear_param.append(param)

rnn_optim = NAS_Adam([lstm_param for lstm_param in rnn.parameters()] + linear_param,
              lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
rnn_criterion = nn.CrossEntropyLoss()


# 方法核心，不断训练RNN控制器，通过训练每一次生成的PANET得到的奖赏值
for trial in range(MAX_TRAIL):
    # 得到RNN controller的序列
    predictions, actions = controller.get_action(state)

    # 将RNN controller得到的序列具现化为PANet模型,并设置optimizer
    scales = controller.get_scales(actions)
    panet = PathAggregation(resnet101, actions, scales)
    panet_optim = Adam(panet.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
    panet_criterion = nn.CrossEntropyLoss()

    # 将数据集放入生成的PANet中训练，返回精度accuracy
    for epoch in range(PANET_NUM_EPOCH):
        for i, (images, labels) in enumerate(train_loader):
            # forward pass
            outputs = panet(images)
            loss = panet_criterion(outputs, labels)

            # backward and optimize
            panet_optim.zero_grad()
            loss.backward()
            panet_optim.step()

            if (i + 1) % 1000 == 0:
                print(
                    'epoch [{}/{}], step [{}/{}], loss:{:.4f}'.format(epoch + 1, PANET_NUM_EPOCH, i + 1, total_step, loss.item()))

    # test the model
    panet.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # images = images.to(device)
            # labels = labels.to(device)
            outputs = panet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print('accuracy of the model on the test images:{:.2f}%'.format(100 * acc))

    # 将精度accuracy进行一定转变，得到本轮生成的PANet奖赏reward
    # compute the reward
    reward = (acc - moving_acc)

    # update moving accuracy with bias correction for 1st update
    if beta > 0.0 and beta < 1.0:
        moving_acc = beta * moving_acc + (1 - beta) * acc
        moving_acc = moving_acc / (1 - beta_bias)
        beta_bias = 0

        reward = np.clip(reward, -0.1, 0.1)

    # 对RNN controller进行后向传播，将奖赏reward作用梯度变化过程中
    one_hots = controller.get_one_hot()
    loss = torch.zeros(1)
    for i in range(len(one_hots)):
        loss += rnn_criterion(outputs, labels)
    loss.backward()


