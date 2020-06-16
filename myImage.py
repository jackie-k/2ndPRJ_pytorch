from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms

trans = transforms.Compose([transforms.Resize((100, 100)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
# trainset = torchvision.datasets.ImageFolder(root="/Users/data-16/Desktop/clothes",
#                                             transform=trans)
trainset = torchvision.datasets.ImageFolder(root="/Users/data16/Desktop/clothes",
                                            transform=trans)

trainset.__getitem__(3)

plt.show()

len(trainset)
print(len(trainset))

classes = trainset.classes
print(classes)

trainloader = DataLoader(trainset,
                         batch_size=16,
                         shuffle=False,
                         num_workers=4)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(labels)


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))

    print(np_img.shape)
    print((np.transpose(np_img, (1, 2, 0))).shape)

    print(images.shape)
    imshow(torchvision.utils.make_grid(images, nrow=4))
    print(images.shape)
    print((torchvision.utils.make_grid(images)).shape)
    print("".join("%5s " % classes[labels[j]] for j in range(16)))
