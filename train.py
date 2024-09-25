import torch
import torchvision.transforms as transforms
import torch.nn as nn

import torch.optim as optim

from dataset import Xray
from model import BaseModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Xray('data', transform, is_train=True)
trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=4, shuffle=True, num_workers=0)

testset = Xray('data', transform, is_train=False)
testloader = torch.utils.data.DataLoader(testset,
                batch_size=4, shuffle=True, num_workers=0)
        
classes = ("COVID19", "NORMAL", "PNEUMONIA")

model = BaseModel().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):
    running_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
    
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 0:
            print()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

path = './Xray_net.pth'
torch.save(model.state_dict(), path)
