import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform=transforms.Compose(
    [
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]
)

classes=('plane','car','bird','cat','deer',
         'dog','frog','horse','ship','truck')

net=LeNet()
save_path='./lenet_cifar10_weight.pth'
net.load_state_dict(torch.load(save_path))

img_path='1.jpg'
img=Image.open(img_path)
img=transform(img)             # [C,H,W]
img=torch.unsqueeze(img,dim=0)  # [N,C,H,W]

with torch.no_grad():
    outputs=net(img)
    predict=torch.max(outputs,dim=1)[1]
    predict=predict.data.numpy()

print(classes[int(predict)])