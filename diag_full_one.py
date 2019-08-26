import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

train_data=torchvision.datasets.MNIST('./', train=True, download=True,transform=torchvision.transforms.ToTensor())
test_data=torchvision.datasets.MNIST('./', train=False, download=True,transform=torchvision.transforms.ToTensor())

train_data_list=[]
train_label_list=[]
for x,y in train_data:
    train_data_list.append(x)
    train_label_list.append(y)
    
test_data_list=[]
test_label_list=[]
for x,y in test_data:
    test_data_list.append(x)
    test_label_list.append(y)
    
train_data_tensor=torch.stack(train_data_list)
train_label_tensor=torch.tensor(train_label_list)
test_data_tensor=torch.stack(test_data_list)
test_label_tensor=torch.tensor(test_label_list)

class Net(nn.Module):
    def __init__(self,feature_dim,q_rank):
        super(Net, self).__init__()
        self.feature_dim=feature_dim
        self.last_weight_dim=feature_dim*10
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, feature_dim)

        self.q_rank=q_rank
        self.prior_mu=torch.tensor(np.zeros(self.last_weight_dim), dtype=torch.float, requires_grad=False)
        self.prior_sigma=torch.tensor(1.0, requires_grad=False)
        
        self.q_mu=(torch.randn(self.last_weight_dim)*0.1).requires_grad_()
        self.q_sigma=torch.tensor(1.0, requires_grad=True)
        self.q_L=(torch.randn(self.last_weight_dim,q_rank)*0.1).requires_grad_()
        
        
        params = list(self.parameters()) + [self.q_mu,self.q_L,self.q_sigma]
        self.optimizer = optim.Adam(params, lr=0.00003)
        self.feature_optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.final_optimizer = optim.Adam([ self.q_mu,self.q_L,self.q_sigma], lr=0.001)

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
        x = torch.tanh( self.fc2(x))
        return x
    
    def predict(self,x,sample_num=100):
        with torch.no_grad():
            final_weight_samples=low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data = self.feature_forward(x,final_weight_sample)
            prediction=(torch.mean(torch.softmax((final_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            return prediction
        
    def test(self,x,label,sample_num=100):
        with torch.no_grad():
            final_weight_samples=low_rank_gaussian_sample(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda(),sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data = self.feature_forward(x,final_weight_sample)
            feature_of_data=self.feature_forward(x)
            prediction=(torch.mean(torch.softmax((inal_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            accuracy=(pred == label).sum().item()/label.size(0)
            return accuracy
        
        
    def predictive_distribution_entropy(self,x):
        with torch.no_grad():
            final_weight_samples=low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)[0]
            output_logit=final_weight_samples@feature_of_data
            output_dis=torch.distributions.categorical.Categorical(logits=output_logit)


            sample=output_dis.sample([100]).reshape([-1,1])
            entropy=-torch.mean(torch.log(torch.mean(torch.exp(output_dis.log_prob(sample)),dim=-1)))
        return entropy
    
    
    def likelihood_eva(self,x,label, sample_num=100, if_online=False):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.last_weight_dim])
            final_weight_samples=low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            output=torch.mean(F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0, 2, 1),dim=-1),0)
            predicted=output.data.max(dim=1, keepdim=True)[1].view(-1)
            accuracy=(predicted == label).sum().item()/label.size(0)
            nll_loss= F.nll_loss(output,label,reduction='sum')
            if if_online:
                curr_prior_mu = self.q_mu.clone().detach()
                curr_prior_L = self.q_L.clone().detach()
                curr_prior_sigma = self.q_sigma.clone().detach()
                kl=KL_low_rank_gaussian_with_low_rank_gaussian(self.q_mu,self.q_L,self.q_sigma,curr_prior_mu,curr_prior_L,curr_prior_sigma)
            else:
                kl=KL_low_rank_gaussian_with_diag_gaussian(self.q_mu,self.q_L,self.q_sigma,self.prior_mu,self.prior_sigma)
            neg_elbo=kl+nll_loss
        
        return -neg_elbo,accuracy
    
    
    
    def predictive_distribution_entropy_2(self,x, sample_num=100):
        with torch.no_grad():
            final_weight_samples=low_rank_gaussian_sample(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda(),sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)[0]
            output_logit=final_weight_samples@feature_of_data
            output_dis=torch.distributions.categorical.Categorical(logits=output_logit)

            final_weight_samples=low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)[0]
            output_logit=final_weight_samples@feature_of_data
            output_dis_for_sample=torch.distributions.categorical.Categorical(logits=output_logit)

            sample=output_dis_for_sample.sample([sample_num]).reshape([-1,1])
            entropy=-torch.mean(torch.log(torch.mean(torch.exp(output_dis.log_prob(sample)),dim=-1)))
        return entropy
    
    
    def predictive_distribution_entropy_batch(self,x):
        with torch.no_grad():
            final_weight_samples=low_rank_gaussian_sample(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda(),sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            output_logit=(final_weight_samples@feature_of_data.t()).permute(2,0,1)
            log_prob=F.log_softmax(output_logit,dim=-1)
            entropy_lower_bound=-torch.mean(torch.sum(log_prob*torch.exp(log_prob),dim=2),dim=1)

        return entropy_lower_bound
    

    
    
    def online_train(self,x,label,sample_num=1):
        train_losses = []
        total_size=x.size(0)
        curr_prior_mu = self.q_mu.clone().detach()
        curr_prior_L = self.q_L.clone().detach()
        curr_prior_sigma = self.q_sigma.clone().detach()
        correct_flag=0
        
#         self.q_mu=(torch.randn(self.last_weight_dim)*0.1).requires_grad_()
#         self.q_sigma=torch.tensor(1.0, requires_grad=True)
#         self.q_L=(torch.randn(self.last_weight_dim,self.q_rank)*0.1).requires_grad_()
        with torch.no_grad():
            feature_of_data_o=self.feature_forward(x)
#         print(feature_of_data.size())
        feature_of_data=feature_of_data_o.clone().detach()
#         while correct_flag<5:
        right=0
        right_list=[]
        for i in range(0,50000):
            self.final_optimizer.zero_grad()
            final_weight_samples=low_rank_gaussian_sample(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda(),sample_num).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            output =F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0,2,1),dim=-1).view(sample_num,10)
            label_batch=label.repeat(sample_num)
            nll_loss= F.nll_loss(output,label_batch,reduction='mean')
            kl=KL_low_rank_gaussian_with_low_rank_gaussian(self.q_mu,self.q_L,self.q_sigma,curr_prior_mu,curr_prior_L,curr_prior_sigma)
            neg_elbo=kl+2*nll_loss
            neg_elbo.backward()
            self.final_optimizer.step()
            train_losses.append(neg_elbo.item())
            if output.data.max(dim=1, keepdim=True)[1][0].item()==label.item():
                right+=1
                
            if i%100==0:
                print('accuracy:', right/100)
                right_list.append(right/100)
                right=0
                
            
#             if output.data.max(dim=1, keepdim=True)[1].item()==label.item():
#                 correct_flag+=1
#             else:
#                 correct_flag=0
        plt.plot(right_list)
        plt.show()
        plt.plot(train_losses)
        plt.show()
        
        
    
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
                final_weight_sample= low_rank_gaussian_one_sample(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda()).view(self.feature_dim,10)
                output = self.forward(x[index],final_weight_sample)
                nll_loss= F.nll_loss(output,label[index],reduction='sum')*(float(x.size(0))/float(batch_size))
                kl=KL_low_rank_gaussian_with_diag_gaussian(self.q_mu.cuda(),self.q_L.cuda(),self.q_sigma.cuda(),self.prior_mu.cuda(),self.prior_sigma.cuda())
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
        plt.plot(train_losses)
        plt.show()
        return train_losses
        
    def test(self):
        correct=0
        for data, target in test_loader:
            pred = self.predict(data)
            correct += pred.eq(target.data.view_as(pred)).sum()
            correct_ratio= float(correct)/len(test_loader.dataset)
        return correct_ratio
    
    
    
nn_tanh = Net(feature_dim=20).cuda()
init_train_data=train_data_tensor[0:10].cuda()
init_train_label=train_label_tensor[0:10].cuda()
accuracy_list=[]
for epoch in range(0,100):
    print('big_epoch:', epoch, 'start training...')
    print('train_data_size',init_train_label.size(0))
    nn_tanh.train(init_train_data,init_train_label)
    
    accuracy=nn_tanh.test(test_data_tensor.cuda(),test_label_tensor.cuda())
    accuracy_list.append(accuracy)
    print('epoch:', epoch, 'test_accuracy', accuracy)
#     plt.title('test_accuracy')
#     plt.plot(accuracy_list)
#     plt.show()
    ### active part
#     print('epoch:', epoch, 'start active learning...')


    for i in range(0,10):
        active_batch_data=train_data_tensor[i*6000:(i+1)*6000].cuda()
        entropy_list=nn_tanh.predictive_distribution_entropy_batch(active_batch_data)
        _, index = entropy_list.max(0)
        init_train_data=torch.cat((init_train_data,active_batch_data[index].view(1,1,28,28).cuda()),0)
        init_train_label=torch.cat((init_train_label,train_label_tensor[index+i*6000].view(-1).cuda()),0)

#     for i in range(0,10):
#         active_batch_data=train_data_tensor[i*6000:(i+1)*6000].cuda()
#         entropy_list=[]
#         for index in range(i*6000,(i+1)*6000):
#             entropy=nn_tanh.predictive_distribution_entropy_2(train_data_tensor[index].cuda())
#             entropy_list.append(entropy)

#         index_max = np.argmax(entropy_list)
#         init_train_data=torch.cat((init_train_data,active_batch_data[index_max].view(1,1,28,28).cuda()),0)
#         init_train_label=torch.cat((init_train_label,train_label_tensor[index_max+i*6000].view(-1).cuda()),0)
        
# plt.title('test_accuracy')
# plt.plot(accuracy_list)
# plt.show()

print(accuracy_list)

np.save('data', init_train_data.cpu().numpy())
np.save('label', init_train_label.cpu().numpy())
