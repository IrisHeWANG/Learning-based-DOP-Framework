import numpy as np
import networkx as nx
import copy
import pandas as pd
import xlwt
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_generator import SynDataset,collate
from model import Net_PGEXTRA,Net_Prox_DGD
from baseline import torch_PGEXTRA,torchProx_DGD,opt_distance,hist_nmse

#########################################################
#                   Trainning Method
#########################################################
def step_loss(x, y, g):
    gamma = g
    n_steps = x.shape[0]
    #print(n_steps)
    di = torch.ones((n_steps)) * gamma
    power = torch.tensor(range(n_steps, 0, -1))
    gamma_a = di ** power
    gamma_a = gamma_a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    y = torch.unsqueeze(y, axis = 0)
    ele_loss = gamma_a * (x - y) **2
    #print(ele_loss.shape)
    #print(torch.mean(ele_loss,  (1,2,3) ))
    loss = torch.mean(ele_loss)
    return loss

#########################################################
#                   LPGEXTRA
#########################################################
num_nodes = 5
num_edges = 6
n = 100
train_num = 1000
test_num = 100
num_epoches = 500

gammas = [0.9]
m_array = [300]
layer_array = [10, 30, 50]
SNR_db_array = [0, 5, 10, 15, 20, 25, 30]

for g in gammas:
  for m in m_array:
  	for num_layers in layer_array:
  		for SNR_db in SNR_db_array:
  			k = m // 5
  			nnz = m //10
  			train_data = SynDataset(train_num)
  			test_data = SynDataset(test_num)
  			train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate)
  			model = Net_PGEXTRA(1e-3, num_layers)
  			optimizer = optim.Adam(model.parameters(), lr=2e-5)
  			model.train()
  			epoch_losses = []
  			for epoch in range(num_epoches):
  			    epoch_loss = 0
  			    for iter, (W, A, y, x_true,pyg_data) in enumerate(train_loader):
  			        z, _,_ = model(W, A, y, pyg_data,num_layers)
  			        loss = step_loss(z, x_true, g)
  			        
  			        optimizer.zero_grad()
  			        loss.backward()
  			        optimizer.step()
  			        epoch_loss += loss.detach().item()
  			    epoch_loss /= (iter + 1)
  
  			val_loader = DataLoader(test_data, batch_size=test_num, shuffle=False, collate_fn=collate)
  
  			for iter, (W, A, y, x_true,pyg_data) in enumerate(val_loader):
  				_,pred,pred_hist = model(W, A, y, pyg_data,num_layers)
  				pred_error = hist_nmse(pred_hist,x_true)
  
  			print('m', m, 'snr', SNR_db ,'layer', num_layers, 'error', pred_error[num_layers])
  
  			    #if(epoch % 10 == 0):
  			    #    print(epoch_loss, model.lam[1], model.step_size[1])
