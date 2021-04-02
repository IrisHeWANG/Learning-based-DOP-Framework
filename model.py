import torch
from torch import nn
import torch.optim as optim
from torch_geometric.utils import from_networkx
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
import torch.nn.functional as F

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
        self.lam = nn.Parameter(torch.ones(num_layers)*step_size*10)
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
            x_2 = self.act(x_32, k)
            
            ret_z.append(x_2)

            x_0 = x_1
            x_1 = x_2
            x_12 = x_32

            k = k + 1
            x_hist.append(x_2)
        
        ret_z = torch.stack(ret_z)
        return ret_z, x_2,x_hist

class Net_Prox_DGD(torch.nn.Module):
    def __init__(self, step_size, num_layers):
        super(Net_Prox_DGD, self).__init__()
        self.step_size = nn.Parameter(torch.ones(num_layers)*step_size)
        self.lam = nn.Parameter(torch.ones(num_layers)*step_size*10)
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
            #x_32 = self.conv(x_1,pyg_data) + x_12 - (self.conv(x_0,pyg_data) + x_0)/2 - \
            #    self.step_size[k] * (self.tgrad_qp(A, b, x_1)-self.tgrad_qp(A, b, x_0))
            x_32 = self.conv(x_1,pyg_data) - self.step_size[k] * self.tgrad_qp(A, b, x_1)
            x_2 = self.act(x_32, k)
            
            ret_z.append(x_2)

            x_0 = x_1
            x_1 = x_2
            x_12 = x_32

            k = k + 1
            x_hist.append(x_2)
        
        ret_z = torch.stack(ret_z)
        return ret_z, x_2,x_hist




