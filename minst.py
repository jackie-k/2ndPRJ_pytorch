import numpy as np
import matplotlib.pyplot as plt
import glob

import torch
from PIL import Image
from torch import nn

from dnnMnist import DNN

device = torch.device("cpu:0")


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


model = DNN().to(device)
model.apply(weights_init)

label_tags = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

for image_path in glob.glob("/Users/data16/Desktop/clothes/*.jpg"):
    print("image_path : {}".format(image_path))
    img = Image.open(image_path).convert("L")
    plt.imshow(img)
    plt.show()

    img = np.resize(img, (1, 784))
    im2arr = ((np.array(img) / 255) - 1) * - 1

    output = model(img)
    _, argmax = torch.max(output, 1)
    pred = label_tags[argmax.item()]

    print(im2arr.shape)
    print(type(im2arr))
    print("prediction : ", pred)


