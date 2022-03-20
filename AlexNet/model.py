import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,class_num=1000,init_weights=False):         # input[3,227,227]
        # 值得注意的一点：原图输入224 × 224，实际上进行了随机裁剪，实际大小为227 × 227。
        super(AlexNet, self).__init__()

        self.conv=nn.Sequential(
            # C1
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=0) ,     # output[96,55,55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),         # output[96,27,27]

            # C2
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2),     # output[256,27,27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),         # output[256,13,13]

            # C3
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),      # outpu[384,13,13]
            nn.ReLU(inplace=True),

            # C4
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),    # output[384,13,13]
            nn.ReLU(inplace=True),

            # C5
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),    # output[256,13,13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),         # output[256,6,6]
        )

        self.fc=nn.Sequential(
            # FC6
            nn.Linear(256*6*6,4096),                                # output[4096]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # FC7
            nn.Linear(4096,4096),                                   # output[4096]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            # FC8
            nn.Linear(4096,class_num)
        )

    def forward(self,x):
        x=self.conv(x)
        x=torch.flatten(x,start_dim=1)  # 第一个维度是batch，所以从第二个维度推平，也就是dim=1的维度
        x=self.fc(x)
        return x

# net=AlexNet()
# inp=torch.randn([1,3,227,227])
# out=net(inp)
# 可以调试查看维度