import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import LeNet
import time
start_time=time.time()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

# 使用cifar10数据集，train：50000张，test：10000张。
# 第一次用的时候，download改为True
trainset = torchvision.datasets.CIFAR10(root='./CIFAR10/train',train=True,
                                        download=False,transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                          shuffle=True,num_workers=0)

# 第一次用的时候，download改为True
testset = torchvision.datasets.CIFAR10(root='./CIFAR10/test',train=False,
                                        download=False,transform=transform)

testloader = torch.utils.data.DataLoader(testset,batch_size=10000,
                                          shuffle=False,num_workers=0)

train_data_iter = iter(trainloader)
train_image,train_label = train_data_iter.__next__()
test_data_iter = iter(testloader)
test_image,test_label = test_data_iter.__next__()

classes=('plane','car','bird','cat','deer',
         'dog','frog','horse','ship','truck')

# # 可视化测试集图片
# def imshow(img):
#     img=img*0.5+0.5     # unnormalize
#     img_np=img.numpy()
#     #转换维度信息[channel,height,width]-->[height,width,channel]
#     plt.imshow(np.transpose(img_np,(1,2,0)))
#     plt.show()
#
#
# print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))

net = LeNet()
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=0.001)

for epoch in range(10):
    epoch_loss=0.0
    for step,data in enumerate(trainloader,start=0):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        epoch_loss+=loss.item()
        # loss是一个一行一列的Tensor，item()用于只取出loss中的数值。
        if step%500==0:
            with torch.no_grad():
                outputs=net(test_image)
                predict=torch.max(outputs,dim=1)[1]
                accuracy=(predict==test_label).sum().item()/test_label.size(0)
                print('Epoch:{:3d},  Step:{:5d},  train loss:{:5.3f},  test accuracy:{:5.3f}'.format(
                      epoch+1,step+1,epoch_loss/500,accuracy))
                epoch_loss=0.0

print("Finish training.")

end_time=time.time()
time_cost=end_time-start_time
print('Time cost:',time_cost)

save_path='./lenet_cifar10_weight.pth'
torch.save(net.state_dict(),save_path)
