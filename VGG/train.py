import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("| Device: ",device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_dir = os.path.join(os.getcwd(), 'data', 'train')
    val_dir = os.path.join(os.getcwd(), 'data', 'val')
    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    val_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)


    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print('| Batch size: ',batch_size)
    print('| Number of workers: ',nw)
    print('| Train num: ',train_num)
    print('| Val num: ',val_num)

    # test_data_iter = iter(val_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))
    model_name='vgg16'
    net = vgg(model_name=model_name,class_num=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './vgg_flower_photos.pth'
    best_acc = 0.0
    start_time=time.time()
    for epoch in range(epochs):
        # train
        net.train()
        epoch_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        epoch_loss_eva=epoch_loss/train_num
        val_accurate=acc/val_num

        print('Epoch:{:3d},  train loss:{:5.3f},  test accuracy:{:5.3f}'.format(
            epoch + 1, epoch_loss_eva, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    end_time=time.time()
    print('| Using time: ' ,end_time-start_time)


if __name__ == '__main__':
    main()