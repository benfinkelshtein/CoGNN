from torch_geometric.utils import get_laplacian, to_dense_adj
import torch 

def get_eigen_decomp(graph, norm):
    laplacian = get_laplacian(graph.edge_index, normalization = norm)
    dense_laplacian = to_dense_adj(laplacian[0], edge_attr = laplacian[1])
    eigenvalues, eigenvectors = torch.linalg.eigh(dense_laplacian)

    return  eigenvalues, eigenvectors