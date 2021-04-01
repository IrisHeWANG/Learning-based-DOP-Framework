import numpy as np
import networkx as nx
import copy
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

def metropolis(adjacency_matrix):
    num_of_nodes = adjacency_matrix.shape[0]
    metropolis=np.zeros((num_of_nodes,num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency_matrix[i,j]==1:
                d_i = np.sum(adjacency_matrix[i,:])
                d_j = np.sum(adjacency_matrix[j,:])
                metropolis[i,j]=1/(1+max(d_i,d_j))
        metropolis[i,i]=1-sum(metropolis[i,:])
    return metropolis

class SynDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.A = []; 
        self.y = []; 
        self.x_true = []
        self.pyg_data=[]
        self.process()
        
        
    def gen_func(self, num_of_nodes, n, m, k):
        A_all = np.random.randn(m, n)
        x = np.random.randn(n)
        x_norm = 0

        while(x_norm < 1e-2):
            x_mask = np.random.rand(n)
            x_mask[x_mask < 1 - nnz/100] = 0
            x_mask[x_mask > 0] = 1
            x_norm = np.linalg.norm(x * x_mask)

        x = x * x_mask
        x = x/np.linalg.norm(x)
        
        SNR = 10**(SNR_db/10)
        
        noise = np.random.randn(m) * np.sqrt(1/SNR)
        y_all = A_all@x + noise

        A = np.zeros((num_of_nodes, k , n))
        y = np.zeros((num_of_nodes, k))
        for ii in range(num_of_nodes):
            start = (k*ii) % m; end = (k*(ii+1) )%m
            if(start > end):
                A[ii,:,:] = np.concatenate((A_all[start:,:],A_all[:end,:]), axis = 0)
                y[ii,:] = np.concatenate((np.expand_dims(y_all[start:], axis = 0), 
                                          np.expand_dims(y_all[:end], axis = 0)), axis = 1)
            else:
                A[ii,:,:] = A_all[start:end,:]
                y[ii,:] = np.expand_dims(y_all[start:end], axis = 0)
                
        x = np.expand_dims(x, axis = 0)
        x = x.repeat(num_of_nodes, axis = 0)
        
        return A, y, x

    def gen_graph(self, num_of_nodes, num_of_edges, directed=False, add_self_loops=True):
        G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
        k = 0
        while (nx.is_strongly_connected(G) if directed else nx.is_connected(G)) == False:
            G = nx.gnm_random_graph(num_of_nodes, num_of_edges, directed=directed)
            k += 1
        # print("Check if connected: ", nx.is_connected(G))
        # nx.draw(G)
        
        edge_index = from_networkx(G).edge_index
        adj = nx.to_numpy_matrix(G)
        return G, adj,edge_index
        
    def process(self):
        _, adj,edge_index = self.gen_graph(num_nodes, num_edges)
        self.edge_index = edge_index
        W = metropolis(adj)
        self.W = [torch.tensor(W, dtype = torch.float)] * self.samples
        
        
        for ii in range(self.samples):
            A, y, x_true = self.gen_func(num_nodes, n, m, k)
            self.A.append(torch.tensor(A, dtype = torch.float) ); 
            self.y.append(torch.tensor(y, dtype = torch.float) ); 
            self.x_true.append(torch.tensor(x_true, dtype = torch.float) )
            
            edge_weight=torch.tensor(W,dtype=torch.float)
            self.pyg_data.append(Data(edge_weight=SparseTensor.from_dense(edge_weight)))        
        
        

    def __getitem__(self, idx):
        return self.W[idx], self.A[idx], self.y[idx], self.x_true[idx], self.pyg_data[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.A)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    W, A, y, x_true, pyg_data = map(list, zip(*samples))
    W = torch.stack(W)
    A = torch.stack(A)
    y = torch.stack(y)
    x_true = torch.stack(x_true)
    pyg_data = Batch.from_data_list(pyg_data)
    return W, A, y, x_true, pyg_data


class MetropolisConv(MessagePassing):
    def __init__(self):
        super(MetropolisConv, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, pyg_data):
        (B, N, D)=x.shape
        out = self.propagate(x=x.view(-1,D), edge_index=pyg_data.edge_weight, node_dim=-1)
        return out.view(B,N,D)

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

class Net_PGEXTRA(torch.nn.Module):
    def __init__(self, step_size, num_layers):
        super(Net_PGEXTRA, self).__init__()
        self.step_size = nn.Parameter(torch.ones(num_layers)*step_size)
        self.lam = nn.Parameter(torch.ones(num_layers)*step_size*5)
        self.num_layers = num_layers
        self.conv=MetropolisConv()
    def tgrad_qp(self, A, b, x):
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
        #print(A.shape, x.shape, b.shape)
        #print(grad_A.shape)
        grad_A = torch.squeeze(grad_A, axis = -1)
        #print(grad_A.shape)
        return grad_A
    
    def act(self, x, ii):
        tau = self.lam[ii] #* self.step_size[ii]
        return F.relu(x - tau) - F.relu( - x - tau)
            
    def forward(self, W, A, b,pyg_data, max_iter):
        (batch_size, num_of_nodes, _, dim) = A.shape
        init_x = torch.zeros((batch_size, num_of_nodes, dim))
        ret_z = []
        
        k = 1
        x_0 = init_x
        x_12 = self.conv(x_0,pyg_data) - self.step_size[0] * self.tgrad_qp(A, b, x_0)
        x_1 = self.act(x_12, 0)
        
        x_hist = [init_x,x_1]
        while (k < max_iter):
            x_32 = self.conv(x_1,pyg_data) + x_12 - (self.conv(x_0,pyg_data) + x_0)/2 - \
                self.step_size[k] * (self.tgrad_qp(A, b, x_1)-self.tgrad_qp(A, b, x_0))
            #x_32 = self.conv(x_1,pyg_data) - self.step_size[k] * self.tgrad_qp(A, b, x_1)
            x_2 = self.act(x_32, k)
            
            ret_z.append(x_2)

            x_0 = x_1
            x_1 = x_2
            x_12 = x_32

            k = k + 1
            x_hist.append(x_2)
        
        ret_z = torch.stack(ret_z)
        return ret_z, x_2,x_hist


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

def opt_distance(opt,x):
    error = 0
    batch_size = x.shape[0]
    num_of_nodes = x.shape[1]
    error = np.linalg.norm(x-opt)**2
    return error/num_of_nodes/batch_size

def hist_nmse(x_hist,opt):
    error = []
    iteration = len(x_hist)
    for k in range(iteration):
        error.append(10*np.log10(opt_distance(x_hist[k].detach(),opt)))
    return error

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
