import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.device('cuda:0')
    if_cuda=True
else:
    device= torch.device('cpu')
    if_cuda=False


class Net(nn.Module):
    def __init__(self,feature_dim):
        super(Net, self).__init__()
        self.feature_dim=feature_dim
        self.final_weight_dim=feature_dim*10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, self.feature_dim)


        self.prior_mu=torch.zeros(self.final_weight_dim, requires_grad=False).to(device)
        self.prior_diag=torch.ones(self.final_weight_dim, requires_grad=False).to(device)

        self.q_mu=torch.randn(self.final_weight_dim, requires_grad=True).to(device)
        self.q_diag=torch.ones(self.final_weight_dim, requires_grad=True).to(device)

        params = list(self.parameters()) + [self.q_mu,self.q_diag]
        self.optimizer = optim.Adam(params, lr=0.00003)
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
            eps=torch.randn([100,self.final_weight_dim]).to(device)
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            prediction=(torch.mean(torch.softmax((inal_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            return prediction


    def test(self,x,label):
        with torch.no_grad():
            eps=torch.randn([100,self.final_weight_dim]).to(device)
            final_weight_samples=(torch.sqrt(self.q_diag.to(device)).repeat(100).view(100,self.final_weight_dim)*eps+self.q_mu.to(device)).view(100,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            pred=(torch.mean(torch.softmax((final_weight_samples@feature_of_data.t()).permute(2, 0, 1),dim=-1),1).data.max(dim=1, keepdim=True)[1]).view(-1)
            accuracy=(pred == label).sum().item()/label.size(0)
            return accuracy



    def predictive_distribution_entropy_batch(self,x, sample_num=100):
        with torch.no_grad():
            eps=torch.randn([sample_num,self.final_weight_dim]) ### 100*200
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu).view(sample_num,20,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)### 70*20

            output_logit=F.log_softmax((final_weight_samples@feature_of_data.t()).permute(2,0,1),dim=-1) ###70*100*10

            eps=torch.randn([sample_num,self.final_weight_dim])
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(sample_num).view(sample_num,self.final_weight_dim)*eps+self.q_mu).view(sample_num,20,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
            output_probs=F.softmax((final_weight_samples@feature_of_data.t()).permute(2,0,1),dim=-1) ###70*100*10
            output_dis_for_sample=sample_from_batch_categorical_multiple_cpu(output_logit,sample_num=30).view(x.size(0),-1) ### 70*100*30
            output_dis_for_sample_one_hot=one_hot_embedding(output_dis_for_sample, 10) ### 70*3000*10
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
            eps=torch.randn([samlpe_num,self.final_weight_dim])
            final_weight_samples=(torch.sqrt(self.q_diag).repeat(samlpe_num).view(samlpe_num,self.final_weight_dim)*eps+self.q_mu).view(samlpe_num,self.feature_dim,10).permute(0, 2, 1)
            feature_of_data=self.feature_forward(x)
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
        for epoch in range(0,3000):
            for it in range(0,int(x.size(0)/batch_size)+1):
                index=np.random.choice(x.size(0),batch_size)
                self.optimizer.zero_grad()
                eps=torch.randn([self.final_weight_dim])
                final_weight_sample= (self.q_mu.to(device)+eps*torch.sqrt(self.q_diag.to(device))).view(self.feature_dim,10)
                output = self.forward(x[index],final_weight_sample)
                nll_loss= F.nll_loss(output,label[index],reduction='sum')*(float(x.size(0))/float(batch_size))
                kl=KL_diag_gaussian(self.q_mu.to(device),self.q_diag.to(device),self.prior_mu.to(device),self.prior_diag.to(device))
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
        return train_losses

if __name__=="__main__":
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

    nn_tanh = Net(feature_dim=20).to(device)
    init_train_data=train_data_tensor[0:1].to(device)
    init_train_label=train_label_tensor[0:1].to(device)

    accuracy_list=[]
    for epoch in range(0,100):
        print('big_epoch:', epoch, 'start training...')
        print('train_data_size',init_train_label.size(0))
        nn_tanh.train(init_train_data,init_train_label)
        accuracy=nn_tanh.test(test_data_tensor.to(device),test_label_tensor.to(device))
        accuracy_list.append(accuracy)
        print('epoch:', epoch, 'test_accuracy', accuracy)
        print(accuracy_list)


        entropy_list=[]
        for i in range(0,10):
           active_batch_data=train_data_tensor[i*6000:(i+1)*6000].to(device)
           entropy_list.extend(nn_tanh.predictive_distribution_entropy_batch(active_batch_data).to_list())

    _, index = entropy_list.max(0)
    init_train_data=torch.cat((init_train_data,train_data_tensor[index].view(1,1,28,28).to(device)),0)
    init_train_label=torch.cat((init_train_label,test_label_tensor[index].view(-1).to(device)),0)
