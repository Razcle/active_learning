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

class diagnet(nn.Module):
    def __init__(self,opt):
        super(diagnet, self).__init__()
        self.feature_dim=opt['feature_dim']
        self.final_weight_dim=opt['feature_dim']*10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, self.feature_dim)
        self.device=opt['device']
        self.if_cuda=opt['if_cuda']


        self.prior_mu=torch.zeros(self.final_weight_dim, requires_grad=False)
        self.prior_diag=torch.ones(self.final_weight_dim, requires_grad=False)

        self.q_mu=torch.randn(self.final_weight_dim, requires_grad=True)
        self.q_diag=torch.ones(self.final_weight_dim, requires_grad=True)

        self.params = list(self.parameters()) + [self.q_mu,self.q_diag]
        self.optimizer = optim.Adam(self.params, lr=opt['optimizer_lr'])
        self.feature_optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.final_optimizer = optim.Adam([ self.q_mu, self.q_diag ], lr=0.001)

    def forward(self, x, final_weight):
        x=x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x =torch.tanh(self.fc2(x))
        x= torch.matmul(x,final_weight)
        return F.log_softmax(x,dim=-1)


    def feature_forward(self, x ):
        x=x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x =  torch.tanh(self.fc2(x))
        return x

    def predict(self,x):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).to(self.device)
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            prediction=(torch.mean(F.softmax((inal_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            return prediction


    def test(self,x,label):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).to(self.device)
            final_weight_samples=(torch.sqrt(self.q_diag.to(self.device)).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu.to(self.device)).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            pred=(torch.mean(F.softmax((final_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            accuracy=(pred == label).sum().item()/label.size(0)
            return accuracy



    def predictive_distribution_entropy_batch(self,x, sample_num=100):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.final_weight_dim]).to(self.device) ### 100*200
            final_weight_samples=(torch.sqrt(self.q_diag.to(self.device)).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.to(self.device)).view(sample_num,20,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)### 70*20

            output_logit=F.log_softmax((final_weight_samples@feature_of_data.t()).permute(2,0,1),dim=-1) ###70*100*10

            eps=torch.randn([sample_num,self.final_weight_dim]).to(self.device)
            final_weight_samples=(torch.sqrt(self.q_diag.to(self.device)).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.to(self.device)).view(sample_num,20,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            output_probs=F.softmax((final_weight_samples@feature_of_data.t()).permute(2,0,1),dim=-1) ###70*100*10
            output_dis_for_sample=sample_from_batch_categorical_multiple(output_logit,sample_num=30,cuda=self.if_cuda).view(x.size(0),-1) ### 70*100*30
            output_dis_for_sample_one_hot=one_hot_embedding(output_dis_for_sample, 10, cuda=self.if_cuda) ### 70*3000*10
            output_probs=output_probs@output_dis_for_sample_one_hot.permute(0,2,1) ### 70*100*3000
            entropy_list=-torch.mean(torch.log(torch.mean(output_probs,dim=1)),dim=-1)
            return entropy_list



    def online_train(self,x,label,samlpe_num=100):
        train_losses = []
        total_size=x.size(0)
        curr_prior_mu = self.q_mu.clone().detach()
        curr_prior_diag= self.q_diag.clone().detach()
        correct_flag=0

        while correct_flag<5:
            self.final_optimizer.zero_grad()
            eps=torch.randn([samlpe_num,self.final_weight_dim]).to(self.device)
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(samlpe_num).view(samlpe_num,self.final_weight_dim)*eps+self.q_mu).view(samlpe_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            output=torch.mean(F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0, 2, 1),dim=-1),0)
            #nll_loss= F.nll_loss(output,label,reduction='sum')
            kl=KL_diag_gaussian(self.q_mu,self.q_diag,curr_prior_mu,curr_prior_diag)
            neg_elbo=kl+nll_loss
            neg_elbo.backward()
            self.final_optimizer.step()
            train_losses.append(neg_elbo.item())


            if output.data.max(dim=1, keepdim=True)[1].item()==label.item():
                correct_flag+=1
            else:
                correct_flag=0
        print(output.data.max(dim=1, keepdim=True)[1].item())
#         plt.plot(train_losses)
#         plt.show()



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
                eps=torch.randn([self.final_weight_dim]).to(self.device)
                final_weight_sample= (self.q_mu.to(self.device)+eps*torch.sqrt(self.q_diag.to(self.device))).view(self.feature_dim,10)
                output = self.forward(x[index],final_weight_sample)
                nll_loss= F.nll_loss(output,label[index])*(float(x.size(0)))
                kl=KL_diag_gaussian(self.q_mu.to(self.device),self.q_diag.to(self.device),self.prior_mu.to(self.device),self.prior_diag.to(self.device))
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
        return train_losses
