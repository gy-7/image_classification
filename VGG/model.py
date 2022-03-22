import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,convs,class_num,init_weights=False):                                             # input[3,224,224]
        super(VGG, self).__init__()
        self.convs=convs        # 根据vgg不同层数，自动编写卷积层。
        self.fcs=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,class_num)
        )
        if init_weights:
            self.initialize_weights()

    def forward(self,x):        # input[batch,3,224,224]
        x=self.convs(x)         # output[batch,512,7,7]
        x=torch.flatten(x,start_dim=1)  # output[batch,512*7*7]
        x=self.fcs(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_convs(cfg:list):     # 根据vgg不同层数，自动定制卷积层
    convs=[]
    in_channels=3
    for out_channles in cfg:
        if out_channles=='m':
            convs+=[nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            convs+=[nn.Conv2d(in_channels,out_channles,kernel_size=3,padding=1),
                    nn.ReLU(True)]
            in_channels=out_channles
    return nn.Sequential(*convs)
    # 以非关键字参数的形式，传入。*convs代表把列表中所有的变量，作为参数，传入到函数nn.Sequential()中

cfgs={
    'vgg11':[64,'m',128,'m',256,256,'m',512,512,'m',512,512,'m'],
    'vgg13':[64,64,'m',128,128,'m',256,256,'m',512,512,'m',512,512,'m'],
    'vgg16':[64,64,'m',128,128,'m',256,256,256,'m',512,512,512,'m',512,512,512,'m'],
    'vgg19':[64,64,'m',128,128,'m',256,256,256,256,'m',512,512,512,512,'m',512,512,512,512,'m'],
}

def vgg(model_name='vgg16',**kwargs):
    assert model_name in cfgs,f"Warning: model number {model_name} not in cfgs dict"
    cfg=cfgs[model_name]

    model=VGG(create_convs(cfg),**kwargs)
    return model