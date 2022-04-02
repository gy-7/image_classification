### **论文原文**：

v1: [ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (arxiv.org)](https://arxiv.org/abs/1704.04861)

v2: [ MobileNetV2: Inverted Residuals and Linear Bottlenecks (arxiv.org)](https://arxiv.org/abs/1801.04381)





### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下，权重文件下载到weights目录下，并重命名，具体看代码中的权重文件名。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



### 训练日志：

```
| Device: cuda:0
| Train dir: E:\Coding\Python\image_classification\MobileNet\data\train
| Train num: 3301
| Val dir: E:\Coding\Python\image_classification\MobileNet\data\val
| Val num: 369
| Bath size: 32
| Number of workers: 4
| Epochs: 10
training.....
Epoch:  1,  train loss:1.356,  test accuracy:0.778
Epoch:  2,  train loss:0.977,  test accuracy:0.827
Epoch:  3,  train loss:0.790,  test accuracy:0.848
Epoch:  4,  train loss:0.681,  test accuracy:0.848
Epoch:  5,  train loss:0.628,  test accuracy:0.851
Epoch:  6,  train loss:0.579,  test accuracy:0.862
Epoch:  7,  train loss:0.545,  test accuracy:0.862
Epoch:  8,  train loss:0.515,  test accuracy:0.864
Epoch:  9,  train loss:0.494,  test accuracy:0.875
Epoch: 10,  train loss:0.466,  test accuracy:0.875
| Time cost: 197.99117422103882
Done.
```



### MobileNet网络结构：

v1:

<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220402134822_mobilenetv1.png" alt="mobilenetv1" style="zoom:33%;" />

v2:

t是扩展因子，c是输出特征图的深度，n是bottleneck重复次数，s是步距（只针对每个bottleneck中的第一个，其余的都为1）。

<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220402134827_mobilenetv2.png" alt="mobilenetv2" style="zoom: 67%;" />







### 普通卷积与DW卷积：

DW卷积就是先让输出特征图的通道数，等于输入特征图的通道数。然后用一个组卷积（一共有输入特征图通道数的组数），然后再用1*1的卷积调整通道数。



#### 普通卷积跟DW卷积(Depthwise)计算量对比：

$X_h$, $X_w$, $X_c$分别代表输入特征图的高，宽，通道数。$K_h$, $K_w$分别代表卷积核的高，宽。

$Y_h$, $Y_w$, $Y_c$分别代表输出特征图的高，宽，通道数。

**普通卷积的计算量**：

![Conv](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220402134805_conv.png)

普通卷积中一条红线的计算量（忽略加法）是：
$$
X_h*X_w*K_h*K_w
$$
一个卷积核共有 $X_c$ 个红线，一共有 $Y_c$ 个卷积核。所以总共的参数量为：
$$
X_h*X_w*K_h*K_w*X_c*Y_c
$$

**DW卷积的计算量**：![DW_conv](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220402134816_dw_conv.png)
$$
(X_h*X_w*K_h*K_w*X_c)+(X_h*X_w*1*1*X_c*Y_c) \\
=X_h*X_w*X_c*(K_h*K_w+Y_c)
$$
我们可以得出DW卷积的计算量远小于普通卷积的计算量，当输出通道数越多的时候，效果越明显。







