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
from diagnet import diagnet
from fullnet import fullnet
from vanillanet import vanillanet

# np.random.seed(0)
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.device('cuda:0')
#     if_cuda=True
# else:
#     device= torch.device('cpu')
#     if_cuda=False


def main(opt):
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

    if opt['net']=='diagnet':
        nn_tanh = diagnet(opt).to(opt['device'])
    elif opt['net']=='fullnet':
        nn_tanh = fullnet(opt).to(opt['device'])
    elif opt['net']=='vanillanet':
        nn_tanh = vanillanet(opt).to(opt['device'])
    init_train_data=train_data_tensor[0:1].to(opt['device'])
    init_train_label=train_label_tensor[0:1].to(opt['device'])

    accuracy_list=[]
    for epoch in range(0,1):
        print('big_epoch:', epoch, 'start training...')
        print('train_data_size',init_train_label.size(0))
       
        if epoch==300:
            nn_tanh.optimizer = optim.Adam(nn_tanh.params, lr=5e-6)


        nn_tanh.train(init_train_data,init_train_label)
        accuracy=nn_tanh.test(test_data_tensor.to(opt['device']),test_label_tensor.to(opt['device']))
        accuracy_list.append(accuracy)
        print('epoch:', epoch, 'test_accuracy', accuracy)
        print(accuracy_list)
        

        entropy_list=[]
        pre_entropy_list=[]
        for i in range(0,1):
            print('iterations',i)
            active_batch_data=train_data_tensor[i*60:(i+1)*60].to(opt['device'])
            entropy_list.extend(nn_tanh.predictive_distribution_entropy_batch(active_batch_data).tolist())
            pre_entropy_list.extend(nn_tanh.preactivation_entropy_batch(active_batch_data).tolist())

        print('entropy_list',entropy_list)
        print('preentropy_list',pre_entropy_list)
        index = np.argmax(entropy_list)
        pre_index = np.argmax(pre_entropy_list)
        print('active label:', index)
        print('pre active label:', pre_index)
        init_train_data=torch.cat((init_train_data,train_data_tensor[index].view(1,1,28,28).to(opt['device'])),0)
        init_train_label=torch.cat((init_train_label,train_label_tensor[index].view(-1).to(opt['device'])),0)

    return accuracy_list
