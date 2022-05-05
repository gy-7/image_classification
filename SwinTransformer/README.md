### 论文原文：

[ Deep Residual Learning for Image Recognition (arxiv.org)](https://arxiv.org/abs/1512.03385)


### 推荐链接：

[CVPR2016:ResNet 从根本上解决深度网络退化问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/106764370)



### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下，权重文件下载到weights目录下，并重命名，具体看代码中的权重文件名。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



### 训练日志：
```
3670 images were found in the dataset.
2939 images for training.
731 images for validation.
Using 6 dataloader workers every process
_IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=['layers.0.blocks.1.attn_mask', 'layers.1.blocks.1.attn_mask', 'layers.2.blocks.1.attn_mask', 'layers.2.blocks.3.attn_mask', 'layers.2.blocks.5.attn_mask', 'layers.2.blocks.7.attn_mask', 'layers.2.blocks.9.attn_mask', 'layers.2.blocks.11.attn_mask', 'layers.2.blocks.13.attn_mask', 'layers.2.blocks.15.attn_mask', 'layers.2.blocks.17.attn_mask'])
[train epoch 1] loss: 0.375, acc: 0.866: 100%|██████████| 92/92 [01:01<00:00,  1.51it/s]
[valid epoch 1] loss: 0.167, acc: 0.943: 100%|██████████| 23/23 [00:05<00:00,  4.02it/s]
[train epoch 2] loss: 0.197, acc: 0.933: 100%|██████████| 92/92 [01:01<00:00,  1.50it/s]
[valid epoch 2] loss: 0.202, acc: 0.929: 100%|██████████| 23/23 [00:05<00:00,  3.98it/s]
[train epoch 3] loss: 0.141, acc: 0.950: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 3] loss: 0.217, acc: 0.934: 100%|██████████| 23/23 [00:05<00:00,  4.00it/s]
[train epoch 4] loss: 0.114, acc: 0.961: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 4] loss: 0.129, acc: 0.958: 100%|██████████| 23/23 [00:05<00:00,  3.99it/s]
[train epoch 5] loss: 0.094, acc: 0.968: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 5] loss: 0.169, acc: 0.955: 100%|██████████| 23/23 [00:05<00:00,  3.97it/s]
[train epoch 6] loss: 0.093, acc: 0.969: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 6] loss: 0.511, acc: 0.880: 100%|██████████| 23/23 [00:05<00:00,  3.98it/s]
[train epoch 7] loss: 0.087, acc: 0.968: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 7] loss: 0.171, acc: 0.952: 100%|██████████| 23/23 [00:05<00:00,  3.96it/s]
[train epoch 8] loss: 0.081, acc: 0.973: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 8] loss: 0.156, acc: 0.953: 100%|██████████| 23/23 [00:05<00:00,  3.96it/s]
[train epoch 9] loss: 0.069, acc: 0.977: 100%|██████████| 92/92 [01:01<00:00,  1.49it/s]
[valid epoch 9] loss: 0.139, acc: 0.960: 100%|██████████| 23/23 [00:05<00:00,  3.95it/s]

进程已结束,退出代码0

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



