论文原文：http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf



#### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下。
+ 运行data目录下的 split_data.py 划分数据集。
+ 运行根目录下 train.py 开始训练，运行 predict.py 测试。



#### 网络结构：

![Alexnet](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220320093555_AlexNet.png)

由当时GPU内存的限制引起的，作者使用两块GPU进行计算，因此分为了上下两部分。目前单GPU就足够了，因此其单GPU的结构图如下所示：

![Alexnet2](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220322131943_new_Alexnet.png)


