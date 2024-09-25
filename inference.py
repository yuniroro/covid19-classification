import torch, code, copy
import torchvision
import torchvision.transforms as transforms
from cnn_finetune import make_model
from dataset import Xray
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = Xray('Xray-Data-resized', transform, is_train=True)
trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=4, shuffle=True, num_workers=0)

testset = Xray('Xray-Data-resized', transform, is_train=False)
testloader = torch.utils.data.DataLoader(testset,
                batch_size=4, shuffle=True, num_workers=0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(1024144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))       
        x = x.view(-1, 1024144)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

path = './Xray_net.pth'

net = Net().cuda()
net.load_state_dict(torch.load(path))

"""
images, labels = dataiter.next()

outputs = net(images.cuda())

_, predicted = torch.max(outputs, 1)
"""

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #code.interact(local = dict(globals(),**locals()))
        correct += (predicted.cpu() == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))

label_list = []
pred_list = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.cuda())
        _, predicted = torch.max(outputs, 1)
        c = (predicted.cpu() == labels).squeeze()

        label_list += [int(i) for i in labels] 
        pred_list += [int(i) for i in predicted.cpu()]

        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = ("COVID19", "NORMAL", "PNEUMONIA")

for i in range(3):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# code.interact(local=locals())

cfm = confusion_matrix(label_list, pred_list)
df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Classification of COVID-19 using X-ray image', fontsize=20)

cfm_plot.figure.savefig("confusion_matrix.png")

cs_report = classification_report(label_list, pred_list, target_names=classes)
print(cs_report)
