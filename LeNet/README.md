### 网络模型结构：

![LeNet模型结构](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220315104051_LeNet.png)



### 文件说明：

| 文件名                   | 用途                 |
| ------------------------ | -------------------- |
| 1.jpg                    | 测试图片             |
| lenet_cifar10_weight.pth | 训练模型后保存的权重 |
| model.py                 | 模型文件             |
| train.py                 | 训练文件             |
| predict.py               | 预测文件             |



### 维度变换：

##### 计算公式：

$$
H'=(H-K+2P)/S+1
$$

##### 常用的层，以及其常用的参数：

三维向量：[channel, height, width]

四维向量:   [batch, channel, height, width]

Conv2d(in_channels, out_channels, kernel_size, stride, padding)

MaxPool2d(kernel_size, stride, padding)

Linear(in_features, out_features)

| input        | layer_name  | layer_parameter | output       |
| ------------ | ----------- | --------------- | ------------ |
| [3, 32, 32]  | Conv2d()    | [3, 16, 5]      | [16, 28, 28] |
| [16, 28, 28] | MaxPool2d() | [2, 2]          | [16, 14, 14] |
| [16, 14, 14] | Conv2d()    | [16, 32, 5]     | [32, 10, 10] |
| [32, 10, 10] | MaxPool2d() | [2, 2]          | [32, 5, 5]   |
| [32, 5, 5]   | view()      | [-1, 32\*5\*5]  | [32\*5\*5]   |
| [32\*5\*5]   | Linear()    | [32\*5\*5, 120] | [120]        |
| [120]        | Linear()    | [120, 84]       | [84]         |
| [84]         | Linear()    | [84, 10]        | [10]         |



### 训练日志：

![LeNet训练日志](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220315130648_LeNet_train_log.png)
