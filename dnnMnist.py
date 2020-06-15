import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

batch_size = 100
num_epochs = 250
learning_rate = 0.0001

# 훈련과 시험에 필요한 데이터 다운로드
root = './MNIST_Fashion'
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5,))])
train_data = dset.FashionMNIST(root=root,
                               train=True, transform=transform, download=True)
test_data = dset.FashionMNIST(root=root,
                              train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


# DNN 클래스의 __init__ 함수로 모델 정의
device = torch.device("cpu:0")


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()

        self.layer1 = nn.Sequential(
            torch.nn.Linear(784, 256, bias=True),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            torch.nn.Linear(256, 64, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            torch.nn.Linear(64, 10, bias=True)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x_out = self.layer1(x)
        x_out = self.layer2(x_out)
        x_out = self.layer3(x_out)
        return x_out


# 맨 처음 가중치
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


model = DNN().to(device)
model.apply(weights_init)  # 맨 처음 가중치 초기화

# 손실 함수
criterion = torch.nn.CrossEntropyLoss().to(device)
# 최소 손실값을 가지는 가중치와 편향값 찾기
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# train 코드 (올바르게 학습이 된다면 에폭이 증가될때마다 손실값은 줄어들게 됨
costs = []
total_batch = len(train_loader)
for epoch in range(num_epochs):
    total_cost = 0

    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cost += loss

    avg_cost = total_cost / total_batch
    print("Epoch:", "%03d" % (epoch+1), "Cost=", "{:.9f}".format(avg_cost))
    costs.append(avg_cost)


# 학습 된 가중치와 편향값을 통해 테스트
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, argmax = torch.max(outputs, 1)
        torch += imgs.size(0)
        correct += (labels == argmax).sum().item()

    print('Accuracy for {} images:{:.2f}%'.format(total, correct / total * 100))


# 정확도를 시작적으로 보기 위해 이미지가 어떻게 분류됐는지 확인 (test data 36개)
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

columns = 6
rows = 6
fig = plt.figure(figsize=(10, 10))

model.eval()
for i in range(1, columns*rows+1):
    data_idx = np.random.randit(len(test_data))
    input_img = test_data[data_idx]
[0].unsqueeze(dim=0).to(device)

output = model(input_img)
_, argmax = torch.max(output, 1)
pred = label_tags[argmax.item()]
label = label_tags[test_data[data_idx][1]]

fig.add_subplot(rows, columns, i)
if pred == label:
    plt.title(pred + 'right!!')
    cmap = 'Blues'
else:
    plt.title('Not' + pred + 'but' + label)
    cmap = 'Reds'
plot_img = test_data[data_idx][0][0,:,:]
plt.imshow(plot_img, cmap=cmap)
plt.axis('off')

plt.show()
