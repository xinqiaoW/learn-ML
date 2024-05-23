# An example in softmax regression
## step 1：import necessary libraries
```
import torch
import torchvision
from torchvision import transforms, datasets
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
import pandas as pd
```
## step 2: load the dataset
```
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
```
## step 3: bulid a neural network and initialize it
```
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def __init__weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


net.apply(__init__weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```
## step 4: choose batch_size and num_epochs
```
batch_size = 256
num_epochs = 10
```
## step 5: process data 
```
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
```
## step 6: train the model
```
for i in range(num_epochs):
    for x_batch, y_batch in train_loader:
        loss_value = loss(net(x_batch), y_batch)
        trainer.zero_grad()
        loss_value.backward()
        trainer.step()
```
## step 7: print results
```
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


all_accuracy = 0
for x_batch, y_batch in test_loader:
    acc = accuracy(net(x_batch), y_batch)
    print("Accuracy on test set: ", acc)
    all_accuracy += acc * len(y_batch)
print("Accuracy on all test set: ", all_accuracy/len(mnist_test))

```
