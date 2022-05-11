论文原文：https://arxiv.org/pdf/1409.1556.pdf

### Pytorch官方预训练权重下载
'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',

'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',

'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',

'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',

'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',

'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',

'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',

'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',

#### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下，也就是跟 split_data.py 同级目录。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



#### 网络结构：

![vgg](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220322051641_vgg.png)



