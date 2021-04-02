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


def tgrad_qp(A, b, x):
    # A: nodes * k * n
    # X: nodes * n
    # Y: nodes * k
    '''grad_A = np.zeros(x.shape)
    for i in range(x.shape[0]):
        grad_A[i] = A[i].T @ (A[i] @ x[i] - b[i])
    return grad_A'''
    x_ = torch.unsqueeze(x, axis = -1)
    b_ = torch.unsqueeze(b, axis = -1)
    
    A_t = A.transpose(2,3)
    grad_A = A_t @ (A @ x_ - b_)
    # print(A.shape, x.shape, b.shape)
    grad_A = torch.squeeze(grad_A, axis = -1)
    return grad_A

def torch_soft(x, tau):
    return F.relu(x - tau) - F.relu( - x - tau)

def opt_distance(x,opt):
    error = 0
    batch_size = x.shape[0]
    num_of_nodes = x.shape[1]
    error = np.linalg.norm(x-opt)**2
    return error/num_of_nodes/batch_size

def hist_nmse(x_hist,opt):
    error = []
    iteration = len(x_hist)
    #print(iteration)
    for k in range(iteration):
        error.append(10*np.log10(opt_distance(x_hist[k].detach(),opt)))
    return error


#########################################################
#                   PGEXTRA
#########################################################


def torch_PGEXTRA(W, A, b, max_iter, step_size,tau):
    (batch_size, num_of_nodes, _, dim) = A.shape
    init_x = torch.zeros((batch_size, num_of_nodes, dim))
    
    
    (batch_size, num_of_nodes, dim) = init_x.shape
    I = torch.unsqueeze(torch.eye(num_of_nodes), axis = 0)
    I = I.repeat(batch_size, 1, 1)
    
    W_hat = (W + I)/2
    
    #initialization
    k = 1
    x_0 = init_x
    x_12 = W @ x_0 - step_size * tgrad_qp(A, b, x_0)
    x_1 = torch_soft(x_12, tau*step_size)
    
    x_hist = [init_x,x_1] #add for plot
    while (k < max_iter):
        
        x_32 = W@x_1 + x_12 - W_hat@x_0 - \
            step_size*(tgrad_qp(A, b, x_1)-tgrad_qp(A, b, x_0))
        x_2 = torch_soft(x_32, tau*step_size)
        
        x_0 = x_1
        x_1 = x_2
        x_12 = x_32
        
        k = k + 1
        
        x_hist.append(x_2)
        
    return x_2,x_hist

#########################################################
#                   Prox-DGD
#########################################################
def torchProx_DGD(W, A, b, max_iter, step_size,tau):
    (batch_size, num_of_nodes, _, dim) = A.shape
    init_x = torch.zeros((batch_size, num_of_nodes, dim))
    
    
    (batch_size, num_of_nodes, dim) = init_x.shape
    I = torch.unsqueeze(torch.eye(num_of_nodes), axis = 0)
    I = I.repeat(batch_size, 1, 1)
    
    W_hat = (W + I)/2
    
    #initialization
    k = 1
    x_0 = init_x
    x_12 = W @ x_0 - step_size * tgrad_qp(A, b, x_0)
    x_1 = torch_soft(x_12, tau*step_size)
    
    x_hist = [init_x,x_1] #add for plot
    while (k < max_iter):
        
        x_32 = W@x_1 -  step_size*tgrad_qp(A, b, x_1)
        x_2 = torch_soft(x_32, tau * step_size)
        
        x_0 = x_1
        x_1 = x_2
        x_12 = x_32
        
        k = k + 1
        
        x_hist.append(x_2)
        
    return x_2,x_hist


