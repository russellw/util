# TODO: reports loss 0 but accuracy < 1
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import linear_code

size = 50


def good(code):
    return linear_code.run(code)


def rands(n):
    pos = []
    neg = []
    while len(pos) < n / 2 or len(neg) < n / 2:
        code = linear_code.rand(size)
        y = good(code)
        if y:
            w = pos
        else:
            w = neg
        if len(w) < n / 2:
            w.append(code)
    w = pos + neg
    random.shuffle(w)
    return w


def convert1(a):
    v = [0.0] * len(linear_code.symbols)
    i = linear_code.symbols.index(a)
    assert i >= 0
    v[i] = 1.0
    return v


def convert(code):
    x = []
    for a in code:
        x.extend(convert1(a))
    y = good(code)
    x = torch.as_tensor(x)
    y = torch.as_tensor([float(y)])
    return x, y


class Dataset1(Dataset):
    def __init__(self, n):
        self.w = [convert(code) for code in rands(n)]

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


batch_size = 8

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(size * len(linear_code.symbols), hidden_size),
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


epochs = 100
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
