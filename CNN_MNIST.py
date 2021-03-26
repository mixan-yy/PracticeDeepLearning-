# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#load the necessary packages
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


# %%
#prepare the data for training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

trainset = datasets.MNIST('MNIST', download = True, train = True, transform = transform )
testset = datasets.MNIST('MNIST', download = False, train = False, transform = transform )
#train, test = random_split(trainset, [55000,5000])
batch_size =  64
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size = batch_size, shuffle = False)


# %%
#see the shape of the images and labels and print some images
import matplotlib.pyplot as plt 
import numpy as np 

dataiter = iter(trainloader)
images, labels = next(dataiter)
plt.imshow(images[0].view(28,28),cmap="gray")
plt.show()
print(labels[0])

print(images.shape)
print(labels.shape)


# %%
# visialize the batch of images together using torchvision
dataiter = iter(trainloader)
images, labels = dataiter.next()
# create grid of images
img_grid = torchvision.utils.make_grid(images)
print(img_grid.shape)
img_grid = np.transpose(img_grid, (1,2,0)) #image.permute(1,2,0) would do the same
print(img_grid.shape)
img_grid = img_grid / 2 + 0.5 #unnormalize 
# show images
plt.imshow(img_grid, cmap='gray')
plt.show()
print(labels)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
#create the model here
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1),
            nn.ReLU(),
            nn.Conv2d(32,64, 3,1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(36864,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10),
            nn.LogSoftmax(dim=1)

        )
    def forward(self, x):
        out = self.net(x).to(device)
        return out


# %%
model = ConvNet().to(device)
print(model)


# %%
#we check our model for one input batch, if everything is in order, if the dimensions are okay
import matplotlib.pyplot as plt 
it = iter(trainloader)
images, labels = next(it)
images = images.to(device)
labels = labels.to(device)
print(model(images).shape)
#plt.imshow(images)


# %%
#define loss function and optimzer type 
from time import time
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr= 0.001, momentum=0.3)
time = time()
epochs = 30


# %%
#train the model here
for epoch in range(epochs):
    correct = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))


# %%
correct = 0
test_loss = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        images = Variable(images).float()
        #print(images.shape)
        output = model.forward(images)

        predicted = torch.max(output, 1)[1]
        correct += (predicted == labels).sum()

    print("Test accuracy:{:.3f}% ".format(100*float(correct) / (len(testloader.dataset))))


# %%



