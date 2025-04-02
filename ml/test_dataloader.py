import random
import statistics

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

size = 5


def good(v):
    return statistics.fmean(v) > 0.5


def rand():
    v = []
    while len(v) < size:
        v.append(random.uniform(0.0, 1.0))
    return v


def rands(n):
    pos = []
    neg = []
    while len(pos) < n / 2 or len(neg) < n / 2:
        v = rand()
        y = good(v)
        if y:
            w = pos
        else:
            w = neg
        if len(w) < n / 2:
            v = torch.as_tensor(v)
            y = torch.as_tensor([float(y)])
            w.append((v, y))
    w = pos + neg
    random.shuffle(w)
    return w


class Dataset1(Dataset):
    def __init__(self, n):
        self.w = rands(n)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


batch_size = 64

train_ds = Dataset1(800)
test_ds = Dataset1(200)

train_dl = DataLoader(train_ds, batch_size=batch_size)
test_dl = DataLoader(test_ds, batch_size=batch_size)

for x, y in train_dl:
    print(x)
    print(x.shape)
    print(x.dtype)
    print(y)
    print(y.shape)
    print(y.dtype)
    break

hidden_size = 100
epochs = 1000


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


device = torch.device("cpu")
model = Net().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


def accuracy(model, ds):
    n = 0
    for x, y in ds:
        y = y[0]
        with torch.no_grad():
            z = model(x)[0]
        if (y and z > 0.5) or (not y and z <= 0.5):
            n += 1
    return n / len(ds)


for epoch in range(epochs):
    for bi, (x, y) in enumerate(train_dl):
        x = x.to(device)
        y = y.to(device)

        loss = criterion(model(x), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % (epochs / 20) == 0 and not bi:
            print(
                "%f\t%f\t%f"
                % (loss, accuracy(model, train_ds), accuracy(model, test_ds))
            )
