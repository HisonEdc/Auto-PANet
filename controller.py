import torch
import torch.nn as nn
import numpy as np
import optimizer
import copy

class StateSpace():
    """
        状态空间类
        方案1：
        属性：候选 规格 候选 规格 候选 候选 候选 候选 候选   候选 规格 候选 规格 候选 候选 候选 候选 候选
        数量： 4   4   5    4   6   7    8   9   10    4   4   5    4   6   7    8   9   10

        方案2：
        属性：候选 候选 规格 候选 候选 规格 候选 候选 候选 候选 候选 候选 候选 候选 候选 候选    候选 候选 规格 候选 候选 规格 候选 候选 候选 候选 候选 候选 候选 候选 候选 候选
        数量： 4   3   4    5   4   4    6   5   7    6   8   7   9    8   10  9       4   3   4    5   4   4    6   5   7    6   8   7   9    8   10  9

        一个问题：
        不同规模的层，是否会对RNN的权重产生质变的影响
    """
    def __init__(self):
        # 感觉这里要设置一个embedding啊
        self.list = [4, 4, 5, 4, 6, 7, 8, 9, 4, 4, 5, 4, 6, 7, 8, 9]
        # self.list = [5, 4, 5, 6, 5, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10]

    def __len__(self):
        return len(self.list)



class RNN(nn.Module):
    """
        RNN 控制器 + mlp
        1.RNN控制器产生序列
        2.进入mlp + softmax产生 候选 和 规格
        2.通过 候选 和 规格 去产生PANet，另外写PANet代码
        3.通过外部的PANet计算精度，通过一定方法得到奖赏reward（可正可负）
        4.对RNN控制器 + mlp后向传播，更新梯度，更新时，通过奖赏对梯度进行作用

        方案1：
        1.一次生成2个候选（选择softmax后最大的2个）
        2.再生成规格

        方案2：
        1.第一次生成1个候选（选择softmax后最大的1个）
        2.第二次再生成1个候选（选择softmax后最大的1个）
        3.再生成规格
    """
    def __init__(self, state_space, input_size=20, hidden_size=32, num_layers=2):
        super(RNN, self).__init__()
        self.state_space = state_space  # 先假设为list类型
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=True)
        self.init_embedding = nn.Embedding(4, input_size)
        # 定义的rnn controller中参数权重为以下所示，其中128是batch_size，自身默认值
        # lstm.weight_ih_l0 torch.Size([128, 20])
        # lstm.weight_hh_l0 torch.Size([128, 32])
        # lstm.bias_ih_l0 torch.Size([128])
        # lstm.bias_hh_l0 torch.Size([128])
        # lstm.weight_ih_l1 torch.Size([128, 32])
        # lstm.weight_hh_l1 torch.Size([128, 32])
        # lstm.bias_ih_l1 torch.Size([128])
        # lstm.bias_hh_l1 torch.Size([128])
        # linear.weight torch.Size([5, 32])
        # linear.bias torch.Size([5])
        self.embeddings = []
        self.linears = []
        self.softmax = nn.Softmax(dim=-1)

        self.predictions = []
        self.actions = []
        for size in self.state_space.list:   # 此处涉及到state_space，不知之后是否需要改
            embedding = nn.Embedding(size, input_size)
            linear = nn.Linear(hidden_size, size, bias=True)
            self.embeddings.append(embedding)
            self.linears.append(linear)


    def forward(self, input):
        output = self.init_embedding(torch.LongTensor([input])).unsqueeze(1)
        predictions = []
        actions = []
        for i in range(len(self.state_space)):
            if i == 0:
                output, (h, c) = self.lstm(output)
            else:
                output, (h, c) = self.lstm(output, (h, c))
            output = self.linears[i](output)
            prediction = self.softmax(output)   # 注意这里的predictions的维度最前面一定要有1
            predictions.append(prediction)
            # if i == 1 or i == 3:
            #     h = (h[0] + h[1]).unsqueeze(0)
            #     c = (c[0] + c[1]).unsqueeze(0)
            # if i == 2 or i == 4:
            #     h = torch.cat((h, h))
            #     c = torch.cat((c, c))
            if i == len(self.state_space) - 1:
                action = list(prediction.detach().squeeze().numpy().argsort()[-2:])
                action = [j + 6 for j in action]
                actions.append(action)
            elif i in [1, 3, 9, 11]:
                action = prediction.detach().squeeze().numpy().argmax()
                output = self.embeddings[i](torch.LongTensor([action])).unsqueeze(1)
                actions.append(action)
            else:
                # 5是FPN中不同尺寸的个数，之后考虑是否可以变成超参
                # 这里要改,这里考虑的只是FPN的搜索（！）
                # 找出prediction中最大的2个数对应的index
                action = list(prediction.detach().squeeze().numpy().argsort()[-2:])
                output = self.embeddings[i](torch.LongTensor(action).unsqueeze(1))
                output = (output[0] + output[1]).unsqueeze(0)
                if i >= 8:
                    action = [j + 6 for j in action]
                actions.append(action)

        self.predictions.append(predictions)
        self.actions.append(actions)
        return predictions, actions

    def initial_weight(self):
        # 初始化RNN控制器+mlp的权重
        pass

    def manage_memory(self):
        # 当self.actions和self.predictions的长度大于20时，将前面的全都删去，减少内存消耗
        if len(self.predictions) > 20:
            self.predictions = self.predictions[-1]
        if len(self.actions) > 20:
            self.actions = self.actions[-1]

class Controller():
    def __init__(self, state_space, rnn, exploration=0.8):
        self.state_space = state_space
        self.exploration = exploration
        self.rnn = rnn
        self.actions = []

    def get_action(self, state):
        """
            按照探索率随机选出一个序列的actions，或者通过RNN controller选出一个序列的actions
            :return: actions
        """
        # 通过RNN controller选出一个序列的actions，需要解决最开始的输入的问题
        # 最开始的输入是否可以是上一轮的output（？）
        # state的定义交由search.py完成，头痛
        predictions, actions = self.rnn(state)

        if np.random.random() < self.exploration:
            print("Generating random action to explore")

            actions = []
            for i in range(len(self.state_space)):
                if i in [1, 3, 9, 11]:
                    action = int(np.random.choice(self.state_space.list[i], size=1))
                    actions.append(action)
                else:
                    action = list(np.random.choice(self.state_space.list[i], size=2))
                    if i >= 8:
                        action = [j + 6 for j in action]
                    actions.append(action)

        self.actions.append(actions)
        return predictions, actions

    def get_one_hot(self, actions):
        # 传入一组actions，将其根据状态空间进行one hot编码
        one_hots = []
        for i in len(actions):
            encode = [0] * self.state_space[i]
            if type(actions[i]) == list:
                for j in actions[i]:
                    encode[j] = 0.5
            else:
                encode[actions[i]] = 1
            one_hots.append(encode)
        return one_hots

    def get_scales(self, actions):
        # 传入一组actions，将其变为scales表示
        scales = copy.deepcopy(actions)
        for i in range(len(scales)):
            if type(scales[i]) == list and i < 8:
                for j in range(len(scales[i])):
                    if scales[i][j] == 4:
                        scales[i][j] = scales[1]
                    elif scales[i][j] == 5:
                        scales[i][j] = scales[3]
                    elif scales[i][j] == 6:
                        scales[i][j] = 3
                    elif scales[i][j] == 7:
                        scales[i][j] = 2
                    elif scales[i][j] == 8:
                        scales[i][j] = 1
            if type(scales[i]) == list and i >= 8:
                for j in range(len(scales[i])):
                    if scales[i][j] == 10:
                        scales[i][j] = scales[9]
                    elif scales[i][j] == 11:
                        scales[i][j] = scales[11]
                    elif scales[i][j] == 6:
                        scales[i][j] = 3
                    elif scales[i][j] == 7:
                        scales[i][j] = 2
                    elif scales[i][j] == 8:
                        scales[i][j] = 1
                    elif scales[i][j] == 9:
                        scales[i][j] = 0
                    elif scales[i][j] == 12:
                        scales[i][j] = 0
                    elif scales[i][j] == 13:
                        scales[i][j] = 1
                    elif scales[i][j] == 14:
                        scales[i][j] = 2
        return scales



