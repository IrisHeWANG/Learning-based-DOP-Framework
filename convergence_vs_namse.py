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


train_num = 1000
test_num = 100
num_layers = 50


train_data = SynDataset(train_num)
val_data = SynDataset(test_num)
test_data = SynDataset(test_num)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, batch_size=100, shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False, collate_fn=collate)



#########################################################
#                   Trainning Method
#########################################################

def step_loss(gamma,x, y):
    #gamma = 0.75
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
print("LPGEXTRA")
model_PGEXTRA = Net_PGEXTRA(1e-3, num_layers)
optimizer = optim.Adam(model_PGEXTRA.parameters(), lr=1e-4)
model_PGEXTRA.train()
epoch_losses = []
for epoch in range(500):
    epoch_loss = 0
    for iter, (W, A, y, x_true,pyg_data) in enumerate(train_loader):
        z, _,_ = model_PGEXTRA(W, A, y, pyg_data,num_layers)
        loss = step_loss(0.83,z, x_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    if(epoch % 10 == 0):
        print(epoch_loss, model_PGEXTRA.lam[1], model_PGEXTRA.step_size[1])

#########################################################
#                   LProx-DGD Trainning
#########################################################
print("LProx-DGD")
model_Prox_DGD = Net_Prox_DGD(1e-3, num_layers)
optimizer = optim.Adam(model_Prox_DGD.parameters(), lr=1e-4)
model_Prox_DGD.train()
epoch_losses = []
for epoch in range(500):
    epoch_loss = 0
    for iter, (W, A, y, x_true,pyg_data) in enumerate(train_loader):
        z, _,_ = model_Prox_DGD(W, A, y, pyg_data,num_layers)
        loss = step_loss(0.93,z, x_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    if(epoch % 10 == 0):
        print(epoch_loss, model_Prox_DGD.lam[1], model_Prox_DGD.step_size[1])



#########################################################
#                   PGEXTRA Trainning
#########################################################
print("PGEXTRA Trainning")
lams = [5e-4,7e-4,1e-3, 2e-3,5e-3,1e-2]
taus = [1e-2, 5e-2,1e-1,5e-1, 1, 5]
best_error = 100
best_pgextra_par = {}
for lam in lams:
    for tau in taus:
        for iter, (W, A, y, x_true,pyg_data) in enumerate(val_loader):
            original,origin_hist = torch_PGEXTRA(W, A, y, 100, lam, tau)
            loss2 = opt_distance(original.detach().numpy(), x_true.numpy())
            loss1 = opt_distance(origin_hist[num_layers].detach().numpy(),x_true.numpy())
            
            print("lamb\ttau\tlayer_loss\t\tfinal_loss")
            print(lam,'\t', tau, '\t',loss1,'\t',loss2)
            
            if loss2 < best_error:
                best_pgextra_par['lam'] = lam
                best_pgextra_par['tau'] = tau
                best_error = loss2



#########################################################
#                   Prox-DGD Trainning
#########################################################
print("Prox-DGD Trainning")
lams = [5e-4,7e-4,1e-3, 2e-3,5e-3]
taus = [1e-2, 5e-2,1e-1,5e-1, 1, 5]
best_error = 100
best_dgd_par = {}
for lam in lams:
    for tau in taus:
        for iter, (W, A, y, x_true,pyg_data) in enumerate(val_loader):
            original,origin_hist = torchProx_DGD(W, A, y, 100, lam, tau)
            loss2 = opt_distance(original.detach().numpy(), x_true.numpy())
            loss1 = opt_distance(origin_hist[num_layers].detach().numpy(),x_true.numpy())
            
            print("lamb\ttau\tlayer_loss\t\tfinal_loss")
            print(lam,'\t', tau, '\t',loss1,'\t',loss2)
            if loss2 < best_error:
                best_dgd_par['lam'] = lam
                best_dgd_par['tau'] = tau
                best_error = loss2

print("Best fpr PGEXTRA:",best_pgextra_par)
print("Best for Prox-DGD:",best_dgd_par)



#########################################################
#                   Test Part
#########################################################
for iter, (W, A, y, x_true,pyg_data) in enumerate(test_loader):
    _,pred_PGEXTRA,pred_PGEXTRA_hist = model_PGEXTRA(W, A, y, pyg_data,num_layers)
    _,pred_DGD,pred_DGD_hist = model_Prox_DGD(W, A, y, pyg_data,num_layers)
    
    original_PGEXTRA,original_PGEXTRA_hist = torch_PGEXTRA(W, A, y, 300,best_pgextra_par['lam'],best_pgextra_par['tau'] )
    original_DGD, original_DGD_hist = torchProx_DGD(W, A, y, 300,best_dgd_par['lam'],best_dgd_par['tau'])


origin_PGEXTRA_error = hist_nmse(original_PGEXTRA_hist,x_true)
origin_DGD_error = hist_nmse(original_DGD_hist,x_true)
pred_PGEXTRA_error = hist_nmse(pred_PGEXTRA_hist,x_true)
pred_DGD_error = hist_nmse(pred_DGD_hist,x_true)

figure_name = "M300"+"NO30"
writer_error=pd.ExcelWriter(figure_name+".xls")
df_error= pd.DataFrame({'PG-EXTRA':origin_PGEXTRA_error,'DGD':origin_DGD_error})
df_error.to_excel(writer_error,sheet_name='Origin')
    
df_feasibility= pd.DataFrame({'PG-EXTRA':pred_PGEXTRA_error,'DGD':pred_DGD_error})
df_feasibility.to_excel(writer_error,sheet_name='GNN')
writer_error.save()  


#########################################################
#                   Plot Part
#########################################################
long_end = 200
x_long = [i for i in range(long_end+1)]
plt.plot(x_long,origin_DGD_error[:long_end+1],linewidth=2,color = 'tab:red')
plt.plot(x_long,origin_PGEXTRA_error[:long_end+1],linewidth=2,color = 'tab:blue' )

x = [i for i in range(num_layers+1)]
plt.plot(x,pred_DGD_error[:num_layers+1],linewidth=2,linestyle='--',color = 'tab:red')
plt.plot(x,pred_PGEXTRA_error[:num_layers+1],linewidth=2,linestyle='--',color = 'tab:blue')
plt.legend(['Prox-DGD','PG-EXTRA','GNN-Prox-DGD','GNN-PG-EXTRA'],loc='upper right',fontsize='large') 
plt.xlabel('iterations',fontsize= 'x-large')
plt.ylabel('NMSE',fontsize= 'x-large')

figure_name = "M300"+"NO30"
plt.savefig(figure_name+".eps")
plt.show()
