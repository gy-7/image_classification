**论文原文**：https://arxiv.org/pdf/1512.03385.pdf



### 推荐链接：

[CVPR2016:ResNet 从根本上解决深度网络退化问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/106764370)



### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



### 训练日志：
```
| Device: cuda:0
| Train dir: E:\Coding\Python\image_classification\ResNet\data\train
| Train num: 3301
| Val dir: E:\Coding\Python\image_classification\ResNet\data\val
| Val num: 369
| Bath size: 16
| Number of workers: 4
training.....
Epoch:  1,  train loss:0.516,  test accuracy:0.908
Epoch:  2,  train loss:0.348,  test accuracy:0.930
Epoch:  3,  train loss:0.293,  test accuracy:0.924
| Time cost: 194.76117491722107
Done.
```


### 残差介绍：

我们想用我们的神经网络拟合一个函数，使得输入数据x，经过函数后得到我们想要的目标值，公式表达也就是：
$$
F(x)=target
$$
已有的神经网络很难拟合的这个映射函数，因此提出拟合残差，而残差学习就是，让我们的神经网络不直接学习 $target$，而是学习 $target-x$ ，公式表达也就是：
$$
F(x)=target-x
$$
拟合残差比拟合映射容易的原因：

[CVPR2016:ResNet 从根本上解决深度网络退化问题](https://zhuanlan.zhihu.com/p/106764370) , 看上边这篇博客中的，**残差结构起作用的原因**，章节。



### 神经网络当时所遇到的问题：

:one: 梯度弥散/爆炸问题，导致模型训练难以收敛。

:two: 网络退化问题，由于非线性激活函数Relu的存在，每次输入到输出的过程都几乎是不可逆的，这也造成了许多不可逆的信息损失。



利用标准初始化和中间层正规化方法，基本可以有效解决梯度弥散/爆炸问题。

利用残差结构，解决网络退化问题。



### 网络结构：

下图中，每一行代表一个stage，一共有5个stage，分别是：conv1, conv2_x, conv3_x, conv4_x, conv5_x。

每个stage里面都是一个block，不同的block有不同的重复次数，例如：50-layer中conv4_x，重复了6次。

![ResNet](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220330034611_resnet.png)

下图，34-layer residual中虚线代表为了维度能够对其，需要对输入的特征进行下采样。

![resnet1](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220330034619_resnet1.png)



