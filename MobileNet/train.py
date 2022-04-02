import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from mobilenet_v2 import MobileNetV2


def main():
    epochs = 10
    best_acc = 0.0
    save_path = './weights/mobilenet_v2_flower.pth'
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("| Device:", device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 加载训练集和验证集
    train_dir = os.path.join(os.getcwd(), 'data', 'train')
    assert os.path.exists(train_dir), "{} path does not exist.".format(train_dir)
    val_dir = os.path.join(os.getcwd(), 'data', 'val')
    assert os.path.exists(train_dir), "{} path does not exist.".format(val_dir)

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    train_steps = len(train_loader)

    validate_dataset = datasets.ImageFolder(root=val_dir,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    # 加载类别信息
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    print("| Train dir:", train_dir)
    print("| Train num:", train_num)
    print("| Val dir:", val_dir)
    print("| Val num:", val_num)
    print("| Bath size:", batch_size)
    print("| Number of workers:", nw)
    print("| Epochs:", epochs)

    net = MobileNetV2(num_classes=5)

    # 加载官网的预训练权重。
    # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    model_weight_path = "./weights/mobilenet_v2.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # 删除分类层classifier的权重
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    print("training.....")
    start_time = time.time()
    for epoch in range(epochs):
        # train
        net.train()
        epoch_loss = 0.0
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        epoch_loss_eva = epoch_loss / train_steps
        val_accurate = acc / val_num

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        print('Epoch:{:3d},  train loss:{:5.3f},  test accuracy:{:5.3f}'.format(
            epoch + 1, epoch_loss_eva, val_accurate))

    end_time = time.time()
    print("| Time cost:", end_time - start_time)
    print('Done.')


if __name__ == '__main__':
    main()
