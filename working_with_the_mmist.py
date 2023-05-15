import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


kwargs = {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST("data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST("data", train=False, transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])),
    batch_size=64, shuffle=True, **kwargs)

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv3 = nn.Conv2d(12, 24, kernel_size=5)
        self.convDropout = nn.Dropout2d()
        self.fcl1 = nn.Linear(320, 60)
        self.fcl2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.convDropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Netz()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    counter = 1
    for batch_id, (data, target) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        counter += 1
        #print(f"epoch:{counter}, loss{loss.data:.3f}")


def test():
    model.eval()
    loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        out = model(data)
        loss += F.nll_loss(out, target, size_average=False).item()
        prediction = out.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).sum()
    loss = loss/len(test_loader.dataset)
    print(f"avg loss: {loss}")
    print(f"prediction: {target}")
    print(f"accuracy: {100*correct/ len(test_loader.dataset)}")


for epoch in range(1, 6):
    train(epoch)
    test()
