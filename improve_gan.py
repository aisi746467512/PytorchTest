#coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np

device=torch.device('cpu')
if os.path.exists('IMgan_images')is False:
    os.mkdir('IMgan_images')
z_dim=100
batch_size=64
learning_rate=0.0002
total_epochs=200

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        layers=[]
        layers.append(nn.Linear(in_features=28*28+10,out_features=512,bias=True))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=256, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=1, bias=True))
        layers.append(nn.Sigmoid())
        self.model=nn.Sequential(*layers)
    def forward(self, x,c):
        x=x.view(x.size(0),-1)
        validity=self.model(torch.cat([x,c],-1))
        return validity
class Generator(nn.Module):
    def __init__(self,z_dim):
        super(Generator,self).__init__()
        layers = []
        layers.append(nn.Linear(in_features=z_dim+10, out_features=128))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=128, out_features=256))
        layers.append(nn.BatchNorm1d(256,0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=256, out_features=512))
        layers.append(nn.BatchNorm1d(512, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(in_features=512, out_features=28*28))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
    def forward(self, z,c):
        x=self.model(torch.cat([z,c],dim=1))
        x=x.view(-1,1,28,28)
        return x


dnet=Discriminator().to(device)
gnet=Generator(z_dim=z_dim).to(device)

bce=nn.BCELoss().to(device)
ones=torch.ones(batch_size).to(device)
zeros=torch.zeros(batch_size).to(device)

g_optimizer=optim.Adam(gnet.parameters(),lr=learning_rate,betas=[0.5,0.999])
d_optimizer=optim.Adam(dnet.parameters(),lr=learning_rate,betas=[0.5,0.999])

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5,))])
dataset=torchvision.datasets.MNIST(root='data/',train=True,transform=transform,download=True)
dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)

fixed_c=torch.FloatTensor(100,10).zero_()
fixed_c=fixed_c.scatter_(dim=1,index=torch.LongTensor(np.array(np.arange(0,10).tolist()*10).reshape([100,1])),value=1)
fixed_c=fixed_c.to(device)
fixed_z=torch.randn([100,z_dim]).to(device)

for epoch in range(total_epochs):
    gnet=gnet.train()
    for i,data in enumerate(dataloader):
        real_images,real_labels=data
        wrong_labels=[]
        for real_label in real_labels:
            tmp=np.arange(10).tolist()
            tmp.remove(float(real_label.data))
            wrong_labels.append(np.random.choice(tmp,1)[0])
        wrong_labels=torch.LongTensor(wrong_labels)

        real_images=real_images.to(device)

        tmp=torch.FloatTensor(real_labels.size(0),10).zero_()
        real_labels=tmp.scatter_(dim=1,index=torch.LongTensor(real_labels.view(-1,1)),value=1)
        real_labels = real_labels.to(device)

        tmp = torch.FloatTensor(wrong_labels.size(0), 10).zero_()
        wrong_labels = tmp.scatter_(dim=1, index=torch.LongTensor(wrong_labels.view(-1, 1)), value=1)
        wrong_labels = wrong_labels.to(device)


        z=torch.randn([batch_size,z_dim]).to(device)
        c=torch.FloatTensor(batch_size,10).zero_()
        c=c.scatter_(dim=1,index=torch.LongTensor(np.random.choice(10,batch_size).reshape([batch_size,1])),value=1)
        c=c.to(device)
        fake_images=gnet(z,c)

        real_loss=bce(dnet(real_images,real_labels),ones)
        fake_loss_1 = bce(dnet(real_images, wrong_labels), ones)
        fake_loss_2=bce(dnet(fake_images.detach(),c),zeros)

        d_loss=real_loss+fake_loss_1+fake_loss_2
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        g_loss=bce(dnet(fake_images,c),ones)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        print("Epoch:%d/%d  Batch:%d/%d -----D_loss:%f   G_loss:%f"%(epoch,total_epochs,i,len(dataloader),d_loss,g_loss))


    gnet=gnet.eval()
    fixed_fake_images=gnet(fixed_z,fixed_c)
    save_image(fixed_fake_images,'IMgan_images/{}.png'.format(epoch),nrow=10,normalize=True)
