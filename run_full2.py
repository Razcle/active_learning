import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from main import *

if __name__=='__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    opt= {}
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        opt['device']= torch.device('cuda:0')
        opt['if_cuda']=True
    else:
        opt['device']= torch.device('cpu')
        opt['if_cuda']=False

    opt['q_rank']=50
    opt['optimizer_lr']=3e-5
    opt['feature_dim']=20
    opt['net']='fullnet'
    result_list=main(opt)
    print(result_list)
    f = open("full_result_anneal.txt", "a")
    f.write("parameters:"+str(opt)+'\n')
    f.write(str(result_list)+'\n')
    f.close()
