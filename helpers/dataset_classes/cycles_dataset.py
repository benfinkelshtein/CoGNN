import torch
from torch_geometric.data import Data
from typing import List
from torch import Tensor


def make_undirected(edge_index: Tensor) -> Tensor:
    edge_index_other_direction = torch.stack((edge_index[1], edge_index[0]), dim=0)
    edge_index = torch.cat((edge_index_other_direction, edge_index), dim=1)
    return edge_index


def create_cycle(max_cycle: int) -> List[Data]:
    data_list = []
    for cycle_size in range(6, max_cycle + 1):
        if cycle_size < (max_cycle + 1 - 6) / 3 + 6:
            train_mask = torch.ones(size=(1,), dtype=torch.bool)
            val_mask = torch.zeros(size=(1,), dtype=torch.bool)
            test_mask = torch.zeros(size=(1,), dtype=torch.bool)
        elif cycle_size < 2 * (max_cycle + 1 - 6) / 3 + 6:
            train_mask = torch.zeros(size=(1,), dtype=torch.bool)
            val_mask = torch.ones(size=(1,), dtype=torch.bool)
            test_mask = torch.zeros(size=(1,), dtype=torch.bool)
        else:
            train_mask = torch.zeros(size=(1,), dtype=torch.bool)
            val_mask = torch.zeros(size=(1,), dtype=torch.bool)
            test_mask = torch.ones(size=(1,), dtype=torch.bool)

        x = torch.ones(size=(cycle_size, 1))
        edge_index1 = torch.tensor([list(range(cycle_size)),
                                    list(range(1, cycle_size)) + [0]])
        edge_index1 = make_undirected(edge_index=edge_index1)
        edge_index2 = torch.tensor([[0, 1, 2] + list(range(3, cycle_size)),
                                    [1, 2, 0] + list(range(4, cycle_size)) + [3]])
        edge_index2 = make_undirected(edge_index=edge_index2)

        data_list.append(Data(x=x, edge_index=edge_index1, y=torch.tensor([0], dtype=torch.long),
                              train_mask=train_mask, val_mask=val_mask, test_mask=test_mask))
        data_list.append(Data(x=x, edge_index=edge_index2, y=torch.tensor([1], dtype=torch.long),
                              train_mask=train_mask, val_mask=val_mask, test_mask=test_mask))

    return data_list


class CyclesDataset(object):

    def __init__(self):
        super().__init__()
        self.data = create_cycle(max_cycle=13)


if __name__ == '__main__':
    data = CyclesDataset()
