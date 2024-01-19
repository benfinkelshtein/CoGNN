from typing import Any
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import dense_to_sparse
from utils.eigen_utils import get_eigen_decomp
from torch_geometric.data import Data
import copy



class AddSpectralBias(BaseTransform):
    def __call__(self, dataset: Any) -> Any:
        new_dataset = []
        for graph in dataset:
            eigenvalues, eigenvectors = get_eigen_decomp(graph, "sym")
            graph.eigenvectors = eigenvectors[0].T
            graph.eigenvalues = eigenvalues
            new_dataset.append(graph)

        return  new_dataset
    
class AddSpectra(BaseTransform):
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))
    def forward(self, data: Data) -> Any:
        eigenvalues, eigenvectors = get_eigen_decomp(data, "sym")
        data.eigenvectors = eigenvectors[0].T
        data.eigenvalues = eigenvalues
        return  data


class GetFirstNEigenvectors(BaseTransform):
    def __init__(self, eigen_dim):
        self.eigen_dim =  eigen_dim  # the number of eigenvectors we wnt to look at

    def __call__(self, dataset: Any) -> Any:
        new_dataset = []
        for graph in dataset:
            graph.Q = []
            for i in range(1,self.eigen_dim):
                graph.Q.append(graph.eigenvectors[i])
            new_dataset.append(graph)
        return  new_dataset

class GetFirstEigen(BaseTransform):
    def __init__(self, eigen_dim):
        self.eigen_dim =  eigen_dim  # the number of eigenvectors we wnt to look at
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))
    
    def forward(self, data: Data):
        data.Q = []
        for i in range(self.eigen_dim):
            data.Q.append(data.eigenvectors[i])
        return data

class GetFirstEigenTresholded(BaseTransform):
    def __init__(self, eigen_dim):
        self.eigen_dim =  eigen_dim  # the number of eigenvectors we wnt to look at
        self.t = 0.1
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))
    
    def forward(self, data: Data):
        data.Q = []
        data_Q_edge = []
        data.Q_thresholded = []
        data.Q_edge_thresholded = []
        for i in range(self.eigen_dim):
            data.Q.append(data.eigenvectors[i])
            dense_Q = data.eigenvectors[i].view(-1,1) @ data.eigenvectors[i].view(1,-1)
            data_Q_edge.append(dense_to_sparse(dense_Q))  # list of (edge_index, edge_weights)
            bool = torch.logical_and(dense_Q > -self.t, dense_Q < self.t)
            dense_Q_thresholded = torch.where(bool, torch.tensor(0.0), dense_Q)
            data.Q_thresholded.append(dense_Q_thresholded)
            data.Q_edge_thresholded.append(dense_to_sparse(dense_Q_thresholded))
        return data
#Bullshit
#graph.Q.append(dense_Q) # list of dense matrices
#dense_Q = graph.eigenvectors[i].view(-1,1) @ graph.eigenvectors[i].view(1,-1)
#graph.Q_edge.append(dense_to_sparse(dense_Q))  # list of (edge_index, edge_weights)
#bool = torch.logical_and(dense_Q > -self.t, dense_Q < self.t)
#dense_Q_thresholded = torch.where(bool, torch.tensor(0.0), dense_Q)
#graph.Q_thresholded.append(dense_Q_thresholded)
#graph.Q_edge_thresholded.append(dense_to_sparse(dense_Q_thresholded))