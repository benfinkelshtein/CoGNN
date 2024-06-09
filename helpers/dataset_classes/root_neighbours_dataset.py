import torch
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, List
from torch import Tensor


class RootNeighboursDataset(object):

    def __init__(self, seed: int, print_flag: bool = False):
        super().__init__()
        self.seed = seed
        self.plot_flag = print_flag
        self.generator = torch.Generator().manual_seed(seed)
        self.constants_dict = self.initialize_constants()

        self._data = self.create_data()

    def get(self) -> Data:
        return self._data

    def create_data(self) -> Data:
        # train, val, test
        data_list = []
        for num in range(self.constants_dict['NUM_COMPONENTS']):
            data_list.append(self.generate_component())
        return Batch.from_data_list(data_list)

    def mask_task(self, num_nodes_per_fold: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = sum(num_nodes_per_fold)
        train_mask = torch.zeros(size=(num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros(size=(num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros(size=(num_nodes,), dtype=torch.bool)

        train_mask[0] = True
        val_mask[num_nodes_per_fold[0]] = True
        test_mask[num_nodes_per_fold[0] + num_nodes_per_fold[1]] = True
        return train_mask, val_mask, test_mask

    def generate_component(self) -> Data:
        data_per_fold, num_nodes_per_fold = [], []
        for fold_idx in range(3):
            data = self.generate_fold(eval=(fold_idx != 0))
            num_nodes_per_fold.append(data.x.shape[0])
            data_per_fold.append(data)

        train_mask, val_mask, test_mask = self.mask_task(num_nodes_per_fold=num_nodes_per_fold)

        batch = Batch.from_data_list(data_per_fold)
        return Data(x=batch.x, edge_index=batch.edge_index, y=batch.y, train_mask=train_mask, val_mask=val_mask,
                    test_mask=test_mask)

    def initialize_constants(self) -> Dict[str, int]:
        return {'NUM_COMPONENTS': 1000, 'MAX_HUBS': 3, 'MAX_1HOP_NEIGHBORS': 10, 'ADD_HUBS': 2, 'HUB_NEIGHBORS': 5,
                'MAX_2HOP_NEIGHBORS': 3, 'NUM_FEATURES': 5}

    def generate_fold(self, eval: bool) -> Data:
        constant_dict = self.initialize_constants()
        MAX_HUBS, MAX_1HOP_NEIGHBORS, ADD_HUBS, HUB_NEIGHBORS, MAX_2HOP_NEIGHBORS, NUM_FEATURES =\
            [constant_dict[key] for key in ['MAX_HUBS', 'MAX_1HOP_NEIGHBORS', 'ADD_HUBS', 'HUB_NEIGHBORS',
                                            'MAX_2HOP_NEIGHBORS', 'NUM_FEATURES']]

        assert MAX_HUBS + ADD_HUBS <= MAX_1HOP_NEIGHBORS
        add_hubs = ADD_HUBS if eval else 0
        num_hubs = torch.randint(1, MAX_HUBS + 1, size=(1,), generator=self.generator).item() + add_hubs
        num_1hop_neighbors = torch.randint(MAX_HUBS + add_hubs, MAX_1HOP_NEIGHBORS + 1, size=(1,),
                                           generator=self.generator).item()
        assert num_hubs <= num_1hop_neighbors

        list_num_2hop_neighbors = torch.randint(1, MAX_2HOP_NEIGHBORS, size=(num_1hop_neighbors - num_hubs,),
                                                generator=self.generator).tolist()
        list_num_2hop_neighbors = [HUB_NEIGHBORS] * num_hubs + list_num_2hop_neighbors

        # 2 hop edge index
        num_nodes = 1  # root node is 0
        idx_1hop_neighbors = []
        list_edge_index = []
        for num_2hop_neighbors in list_num_2hop_neighbors:
            idx_1hop_neighbors.append(num_nodes)
            if num_2hop_neighbors > 0:
                clique_edge_index = torch.tensor([[0] * num_2hop_neighbors, list(range(1, num_2hop_neighbors + 1))])
                # clique_edge_index = torch.combinations(torch.arange(num_2hop_neighbors), r=2).T
                list_edge_index.append(clique_edge_index + num_nodes)

            num_nodes += num_2hop_neighbors + 1

        # 1 hop edge index
        idx_0hop = torch.tensor([0] * num_1hop_neighbors)
        idx_1hop_neighbors = torch.tensor(idx_1hop_neighbors)
        hubs = idx_1hop_neighbors[:num_hubs]
        list_edge_index.append(torch.stack((idx_0hop, idx_1hop_neighbors), dim=0))
        edge_index = torch.cat(list_edge_index, dim=1)

        # undirect
        edge_index_other_direction = torch.stack((edge_index[1], edge_index[0]), dim=0)
        edge_index = torch.cat((edge_index_other_direction, edge_index), dim=1)

        # features
        x = 4 * torch.rand(size=(num_nodes, NUM_FEATURES), generator=self.generator) - 2

        # labels
        y = torch.zeros_like(x)
        y[0] = torch.mean(x[hubs], dim=0)
        return Data(x=x, edge_index=edge_index, y=y)


if __name__ == '__main__':
    data = RootNeighboursDataset(seed=0, print_flag=True)
