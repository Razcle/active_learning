import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tools import *
import operator
import itertools

class vanillanet(nn.Module):
    def __init__(self,opt):
        super(vanillanet, self).__init__()
        self.feature_dim=opt['feature_dim']
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, self.feature_dim)
        self.fc3 = nn.Linear(self.feature_dim,10)
        self.device=opt['device']
        self.if_cuda=opt['if_cuda']

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x=x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x =torch.tanh(self.fc2(x))
        x= self.fc3(x)
        return F.log_softmax(x,dim=-1)

    def predictive_distribution_entropy_batch(self,x):
        with torch.no_grad():
            batch_logit=self.forward(x)
            batch_probs=torch.exp(batch_logit)
            batch_entropy=-torch.sum(batch_logit*batch_probs,dim=-1)
#             print(batch_entropy.size())
        return batch_entropy



    def train(self,x,label):
        train_losses = []
        if x.size(0)<100:
            batch_size=x.size(0)
            iteration=1
        else:
            batch_size=100
            iteration=int(x.size(0)/batch_size)
        for epoch in range(0,3000):
            for it in range(0,iteration):
                index=np.random.choice(x.size(0),batch_size)
                self.optimizer.zero_grad()
                output = self.forward(x[index])
                nll_loss= F.nll_loss(output,label[index])
                nll_loss.backward()
                self.optimizer.step()
                train_losses.append(nll_loss.item())
#         plt.title('training_accuracy')
#         plt.plot(train_losses)
#         plt.show()
        return train_losses

    def test(self,x,label):
        correct=0
        pred = (self.forward(x).data.max(dim=1, keepdim=True)[1]).view(-1)
#         print(pred)
#         print(label)
#         print(torch.nonzero(pred-label))
        accuracy=(pred == label).sum().item()/label.size(0)
        return accuracy
