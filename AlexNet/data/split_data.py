import os
from shutil import copy, rmtree
import random
import glob

def mk_file(dir: str):
    if os.path.exists(dir):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(dir)
    os.makedirs(dir)


def main():
    # 训练集：测试集  9：1
    train_rate = 0.9

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "flower_photos")
    assert os.path.exists(data_dir), "path '{}' does not exist.".format(data_dir)

    flower_class = [cla for cla in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, cla))]  # 只保留文件夹，不包含文件

    # 建立训练集的文件夹
    train_dir = os.path.join(cwd, "train")
    mk_file(train_dir)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_dir, cla))

    # 建立验证集的文件夹
    test_dir = os.path.join(cwd, "val")
    mk_file(test_dir)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(test_dir, cla))

    for cla in flower_class:
        cla_path = os.path.join(data_dir, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        val_index = random.sample(images, k=int(num*train_rate))
        for image in images:
            original_path = os.path.join(cla_path, image)

            if image in val_index:
                # 将分配至验证集中的文件复制到相应目录
                new_path = os.path.join(train_dir, cla)
            else:
                # 将分配至训练集中的文件复制到相应目录
                new_path = os.path.join(test_dir, cla)

            copy(original_path, new_path)

    train_num=sum([len(glob.glob(train_dir+'\\'+cla+'\\*.jpg')) for cla in flower_class])
    test_num=sum([len(glob.glob(test_dir+'\\'+cla+'\\*.jpg')) for cla in flower_class])
    print('| Train num: ',train_num)
    print('| Test num: ',test_num)
    print("Done!")

if __name__ == '__main__':
    main()
