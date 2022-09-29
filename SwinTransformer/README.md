### 代码来源于 ：

[deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

B站网络导师了，强烈推荐。



### 论文和开源库：

[ Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (arxiv.org)](https://arxiv.org/abs/2103.14030)

https://github.com/microsoft/Swin-Transformer 权重在这里下载


### 运行步骤：

+ 点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
+ 数据集下载后解压到data目录下，运行data文件夹下split_data.py脚本。
+ 新建一个weights文件夹，权重文件下载到weights目录下，并重命名，具体看代码中的权重文件名。
  + 权重下载速度慢：可以用这个网址转链：[GitHub Proxy 代理加速 (ghproxy.com)](https://ghproxy.com/)
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



### 网络结构：

![image-20220505220529148](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220505141705_image-20220505220529148.png)

> Swin-T: C = 96, layer numbers = {2,2,6,2}
>
> Swin-S: C = 96, layer numbers ={2,2,18,2}
>
> Swin-B: C = 128, layer numbers ={2,2,18,2}
>
> Swin-L: C = 192, layer numbers ={2,2,18,2}



### 多尺度输入：

Swin Transformer输入图片尺寸也应该是固定的，因为在Shift Windows过程中，生成的Mask attention尺寸是固定的，因为我们要让生成的Mask attention符合我们的窗口尺寸以及窗口数量。

所以如果想要支持动态的输入尺寸，我们就必须动态地生成这些Mask attention。



### 下面放一组图片便于代码理解，图片均来源于

###  [12.2 使用Pytorch搭建Swin-Transformer网络\_哔哩哔哩\_bilibili](https://www.bilibili.com/video/BV1yg411K7Yc/?spm_id_from=333.788)



#### Patch Partition + Linear Embedding:

![Patch Partition Linear Embedding:](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220505143343_image-20220505222857683.png)

#### Patch Merging

![Patch Merging](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220505143350_image-20220505223031262.png)

#### W-MSA: Windows Multi-head Self-Attention

![W-MSA](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220505143355_image-20220505223107698.png)

#### SW-MSA: Shifted Windows Multi-head Self-Attention

![SW-MSA](https://images.cnblogs.com/cnblogs_com/blogs/471668/galleries/1907323/o_220505143400_image-20220505223200788.png)



