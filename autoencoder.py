# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#imports
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
from torchvision.utils import save_image



# %%
#Among other use cases autoencoders are used for data denoising and dimensionality reduction for data visualization. 


# %%
#x is the image tensor. We need to reshape it to see as an image. Before that we will increase the pixel values by 0.5. Then clamp the values min = 0, max = 1. This will make the digits brighter, and rest will be darker. We can print x without processing this way. This will lead to darker, unrecognisible digits. 
def to_image(x):
    x = x+0.5
    x = x.clamp(0,1)
    x = x.view(x.size(0),1,28,28)
    return x


# %%
#here we will save the output
if not os.path.exists('./decoded_images'):
    os.mkdir('./decoded_images')


# %%
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5],[0.5])
])


# %%
dataset = datasets.MNIST(root = './MNIST', download=False, transform=transform)


# %%
batch_size = 128
num_epochs = 30


# %%
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle  = True)


# %%
#define the model here
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,12)
        )
        self.decoder = nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()

        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
model = Autoencoder().to(device)


# %%
#corss entropy loss doesn't work for this particular problem. 
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 


# %%
loss_list = list()
for epoch in range(1,num_epochs+1):
    losses = 0
    for img, _ in dataloader:
        img = img.view(-1,28*28)
        img = img.to(device)
        model_out = model(img).to(device)
        loss = criterion(model_out,img) #comparison between model output and original images
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss
    print("Epoch: ", epoch, " Loss: ", losses)
    loss_list.append(losses)
    if epoch%10 == 0: #we will print image every 10 epochs
        pic = to_image(model_out.cpu().data)
        save_image(pic, './decoded_images/image_{}.png'.format(epoch))


# %%
import matplotlib.pyplot as plt
epoch = [i for i in range(1,num_epochs+1)]
plt.plot(epoch, loss_list)
plt.show()


