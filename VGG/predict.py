import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "rose.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model_name='vgg16'
    net = vgg(model_name=model_name,class_num=5, init_weights=True)

    # load model weights
    weights_path = "./vgg_flower_photos.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path))

    net.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(net(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "| Class: {}   Prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("Class: {:10}   Prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()
