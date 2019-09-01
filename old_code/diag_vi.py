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
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)

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
    def __init__(self,feature_dim):
        super(Net, self).__init__()
        self.feature_dim=feature_dim
        self.final_weight_dim=feature_dim*10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, self.feature_dim)
        
        
#         self.prior_mu=torch.zeros(200, requires_grad=False).cuda()
#         self.prior_diag=torch.ones(200, requires_grad=False).cuda()
        
#         self.q_mu=torch.randn(200, requires_grad=True).cuda()
#         self.q_diag=torch.ones(200, requires_grad=True).cuda()

        self.prior_mu=torch.tensor(np.zeros(self.final_weight_dim), dtype=torch.float, requires_grad=False)
        self.prior_diag=torch.tensor(np.ones(self.final_weight_dim), dtype=torch.float, requires_grad=False)
        
        self.q_mu=(torch.randn(self.final_weight_dim)*0.1).requires_grad_()
        self.q_diag=torch.tensor(np.ones(self.final_weight_dim), dtype=torch.float, requires_grad=True)
    
        self.params = list(self.parameters()) + [self.q_mu,self.q_diag]
        self.optimizer = optim.Adam(self.params, lr=0.00005)
        self.feature_optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.final_optimizer = optim.Adam([ self.q_mu, self.q_diag ], lr=0.001)

    def forward(self, x, final_weight):
        x=x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x =self.fc2(x)
        x= torch.matmul(x,final_weight)
        return F.log_softmax(x,dim=-1)
    
    
    def feature_forward(self, x ):
        x=x.view(-1,1,28,28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x =  self.fc2(x)
        return x
    
    def predict(self,x):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).cuda()
            ##100 10 70
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            
            prediction=(torch.mean(torch.softmax((inal_weight_samples@feature_of_data).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            return prediction
        
        
    def test(self,x,label):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).cuda()
            ##100 10 70
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu.cuda()).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
#             print(final_weight_samples.size())
#             print(x.size())
#             print(feature_of_data.size())
            pred=(torch.mean(torch.softmax((final_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            #print(pred.size())
            accuracy=(pred == label).sum().item()/label.size(0)
            return accuracy

        

        
        
    def predictive_distribution_entropy(self,x):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).cuda()
            #eps=torch.tensor(np.random.normal(size=[100,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu.cuda()).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            #100,10,20*20,70
            output_logit=final_weight_samples@feature_of_data.t()
            output_dis=torch.distributions.categorical.Categorical(logits=output_logit)

            sample=output_dis.sample([100]).reshape([-1,1])
            entropy=-torch.mean(torch.log(torch.mean(torch.exp(output_dis.log_prob(sample)),dim=-1)))
            return entropy

#     def likelihood_eva(self,x, label, sample_num=100, if_online=False):
#         with torch.no_grad():
#             eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
#             final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
#             feature_of_data=self.feature_forward(x)
#             output=torch.mean(F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0, 2, 1),dim=-1),0)
#             nll_loss= F.nll_loss(output,label,reduction='sum')
#             neg_elbo=kl+nll_loss

#             return -neg_elbo
        
#     def likelihood_eva(self,x,label, sample_num=100, if_online=False):
#         with torch.no_grad():
#             eps=torch.randn([sample_num,self.final_weight_dim])
#             final_weight_samples=(torch.sqrt(self.q_diag).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
#             feature_of_data=self.feature_forward(x)
#             print(final_weight_samples.size())
#             output=torch.mean(F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0, 2, 1),dim=-1),0)

#             predicted=output.data.max(dim=1, keepdim=True)[1].view(-1)
#             accuracy=(predicted == label).sum().item()/label.size(0)

#             print(output.size())
#             nll_loss= F.nll_loss(output,label,reduction='sum')
#             if if_online:
#                 curr_prior_mu = self.q_mu.clone().detach()
#                 curr_prior_diag= self.q_diag.clone().detach()
#                 kl=KL_diag_gaussian(self.q_mu,self.q_diag,curr_prior_mu,curr_prior_diag)
#             else:
#                 kl=KL_diag_gaussian(self.q_mu,self.q_diag,self.prior_mu,self.prior_diag)
#             neg_elbo=kl+nll_loss

#             return -neg_elbo,accuracy
    
    def predictive_distribution_entropy_lower_bound(self,x, sample_num=100):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
            #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)

            output_logit=(final_weight_samples@feature_of_data.t()).permute(2, 0, 1)
            y_samples=sample_from_batch_categorical(output_logit).view(-1)
#             print(y_samples.size())
            output=torch.mean(F.log_softmax(output_logit,dim=-1),1)
#             print(output.size())
            output_repeat=output.repeat(1,sample_num).view(-1,10)
            nll= F.nll_loss(output_repeat,y_samples,reduction='mean')
            kl=KL_diag_gaussian(self.q_mu.cuda(),self.q_diag.cuda(),self.prior_mu.cuda(),self.prior_diag.cuda())
            return nll+kl
    
    def predictive_distribution_entropy_2(self,x, sample_num=100):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
            #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)[0]

            output_logit=final_weight_samples@feature_of_data
            output_dis=torch.distributions.categorical.Categorical(logits=output_logit)

            eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
            #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)[0]
            output_logit=final_weight_samples@feature_of_data
            output_dis_for_sample=torch.distributions.categorical.Categorical(logits=output_logit)

            sample=output_dis_for_sample.sample([sample_num]).reshape([-1,1])
            entropy=-torch.mean(torch.log(torch.mean(torch.exp(output_dis.log_prob(sample)),dim=-1)))
            return entropy
        
        
    def predictive_distribution_entropy_batch(self,x, sample_num=100):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
            #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
#             print(final_weight_samples.size())
            output_logit=(final_weight_samples@feature_of_data.t()).permute(2,0,1)
            log_prob=F.log_softmax(output_logit,dim=-1)
            entropy_lower_bound=-torch.mean(torch.sum(log_prob*torch.exp(log_prob),dim=2),dim=1)
#             print(entropy_lower_bound.size())
            
            return entropy_lower_bound
            
            
            
            
            
#             output_dis=torch.distributions.categorical.Categorical(logits=output_logit)

#             eps=torch.randn([sample_num,self.final_weight_dim]).cuda()
#             #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
#             final_weight_samples=(torch.sqrt(self.q_diag.cuda()).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu.cuda()).view(sample_num,20,10).permute(0, 2, 1)
#             feature_of_data=self.feature_forward(x)
#             output_logit=feature_of_data@final_weight_samples
#             output_dis_for_sample=torch.distributions.categorical.Categorical(logits=output_logit)

#             sample=output_dis_for_sample.sample([sample_num]).reshape([-1,1])
#             entropy=-torch.mean(torch.log(torch.mean(torch.exp(output_dis.log_prob(sample)),dim=-1)))
#             return entropy
    

    
    
    def online_train(self,x,label,samlpe_num=100):
        train_losses = []
        total_size=x.size(0)
        curr_prior_mu = self.q_mu.clone().detach()
        curr_prior_diag= self.q_diag.clone().detach()
        correct_flag=0

        while correct_flag<5:
            
#             lr = 0.01 * (0.1 ** (it // 2000))
#             for param_group in self.final_optimizer.param_groups:
#                 param_group['lr'] = lr
        
            self.final_optimizer.zero_grad()
            eps=torch.randn([samlpe_num,self.final_weight_dim])
            #eps=torch.tensor(np.random.normal(size=[sample_num,200]),dtype=torch.float)
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(samlpe_num).view(samlpe_num,self.final_weight_dim)*eps+self.q_mu).view(samlpe_num,self.feature_dim,10).permute(0, 2, 1)

    #         eps=torch.tensor(np.random.normal(size=[200]),dtype=torch.float)
    #         final_weight_samples= (self.q_mu+eps*torch.sqrt(self.q_diag)).view(20,10)
            feature_of_data=self.feature_forward(x)
    #         print(feature_of_data.size())
            output=torch.mean(F.log_softmax((final_weight_samples@feature_of_data.t()).permute(0, 2, 1),dim=-1),0)
            
            nll_loss= F.nll_loss(output,label,reduction='sum')
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
        #scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        for epoch in range(0,5000):
            for it in range(0,int(x.size(0)/batch_size)+1):
                index=np.random.choice(x.size(0),batch_size)
                self.optimizer.zero_grad()

    #             conditional_loss=0
    #             for i in range(0,10):
                eps=torch.randn([self.final_weight_dim]).cuda()
                final_weight_sample= (self.q_mu.cuda()+eps*torch.sqrt(self.q_diag.cuda())).view(self.feature_dim,10)
                output = self.forward(x[index],final_weight_sample)
                nll_loss= F.nll_loss(output,label[index],reduction='sum')*(float(x.size(0))/float(batch_size))
                kl=KL_diag_gaussian(self.q_mu.cuda(),self.q_diag.cuda(),self.prior_mu.cuda(),self.prior_diag.cuda())
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
            #scheduler.step()
#         plt.plot(train_losses)
#         plt.show()
        
        
#     def test(self):
#         correct=0
#         for data, target in test_loader:
#             pred = self.predict(data)
#             correct += pred.eq(target.data.view_as(pred)).sum()
#             correct_ratio= float(correct)/len(test_loader.dataset)
#         return correct_ratio
    
    
    
nn_tanh = Net(feature_dim=20).cuda()
init_train_data=train_data_tensor[0:10].cuda()
init_train_label=train_label_tensor[0:10].cuda()
accuracy_list=[]
for epoch in range(0,100):
    print('big_epoch:', epoch, 'start training...')
    print('train_data_size',init_train_label.size(0))
    nn_tanh.train(init_train_data,init_train_label)
   # learning_rate=0.0005-(0.0005-0.00003)*(np.exp(epoch-100))
    #nn_tanh.optimizer = optim.Adam(nn_tanh.params, lr=learning_rate)
    
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
