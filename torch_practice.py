import torch
import torch.nn as nn


inputs = torch.Tensor(1,1,28,28)
print('size of Tensor : {}'.format(inputs.shape))

conv1 = nn.Conv2d(1,32,3,padding=1)
print(conv1)

conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
print(conv2)

pool = nn.MaxPool2d(2)
print(pool)

out = conv1(inputs)
print(out.shape)

out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

out.size(0)
out.size(1)
out.size(2)
out.size(3)

out = out.view(out.size(0), -1)
print(out.shape)

fc = nn.Linear(3136,10)
out = fc(out)
print(out.shape)


