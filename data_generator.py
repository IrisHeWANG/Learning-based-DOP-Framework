import numpy as np
import networkx as nx
import torch
from torch import nn
import torch.optim as optim
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul

num_nodes = 5
num_edges = 6
n = 100
m = 300
k = 60
nnz = 30

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
        
        SNR_db = 30
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


