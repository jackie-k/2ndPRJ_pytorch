import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import logging
import optparse
import random
import sys


LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debut': logging.DEBUG}

def init():
    parser = optparse.OptionParser()
    parser.add_option('-l', '--logging-level', help='Logging level')
    parser.add_option('-f', '--logging-file', help='Logging file name')
    (options, args) = parser.parse_args()
    logging_level = LOGGING_LEVELS.get(options.logging_level, logging.NOTSET)
    logging.basicConfig(level=logging_level, filename=options.logging_file,
                        format='%asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

print(sys.executable)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)

    learning_rate = 0.001

    training_epochs = 15
    batch_size = 100

    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    minst_test = dsets.MNIST(root='MNIST_data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)


    class SoftmaxClassifierModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(784.10)

        def forward(self, x):
            return self.linear(x)

    linear = SoftmaxClassifierModel().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(data_loader)

        for X, Y in data_loader:
            X = X.view(-1, 28 * 28).to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            h = linear(X)
            cost = criterion(h, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print('Learning finished')
