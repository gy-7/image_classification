### **论文原文**：

[ Aggregated Residual Transformations for Deep Neural Networks (arxiv.org)](https://arxiv.org/abs/1611.05431)



### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下，权重文件下载到weights目录下，并重命名，具体看代码中的权重文件名。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



### 训练日志：

```
| Device: cuda:0
| Train dir: E:\Coding\Python\image_classification\ResNeXt\data\train
| Train num: 3301
| Val dir: E:\Coding\Python\image_classification\ResNeXt\data\val
| Val num: 369
| Bath size: 4
| Number of workers: 4
training.....
Epoch:  1,  train loss:0.774,  test accuracy:0.873
Epoch:  2,  train loss:0.606,  test accuracy:0.873
Epoch:  3,  train loss:0.575,  test accuracy:0.881
Epoch:  4,  train loss:0.526,  test accuracy:0.886
Epoch:  5,  train loss:0.482,  test accuracy:0.900
Epoch:  6,  train loss:0.455,  test accuracy:0.894
Epoch:  7,  train loss:0.447,  test accuracy:0.892
Epoch:  8,  train loss:0.398,  test accuracy:0.932
Epoch:  9,  train loss:0.410,  test accuracy:0.881
Epoch: 10,  train loss:0.399,  test accuracy:0.908
| Time cost: 2377.1561698913574
Done.
```



### 网络结构：

主要改变就是换成了组卷积，groups conv。

<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220401045418_ResNeXt.png" alt="ResNeXt" style="zoom: 33%;" />

原先是1*1的卷积核，共有64个卷积核，每个卷积核和都要与输入的特征图进行卷积。

现在是1*1的卷积核，共有128个卷积核，分为32个组，每个组有四个卷积核。每个组只跟输入的特征图中对应的地方进行卷积，而不会对整个特征图进行卷积，这样大大降低了参数量。



<img src="https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220401045426_groups_conv.png" alt="resnext" style="zoom: 50%;" />



原先的参数量：

$$
h*w*c_1*c_2
$$

使用分组卷积后的参数量：

$$
h*w*(\frac{c_1}{g})*(\frac{c_2}{g})*g \\
=h*w*c_1*c_2*\frac{1}{g}
$$

h：卷积核高度

w：卷积核宽度

c1：输入特征图通道数

c2：输出特征图通道数

g：分组卷积的组数
