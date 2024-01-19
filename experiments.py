from argparse import Namespace
import torch
import sys
import tqdm
from typing import Tuple, Any
from torch_geometric.loader import DataLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
import numpy as np

from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
from helpers.metrics import LossesAndMetrics
from helpers.utils import set_seed
from models.CoGNN import CoGNN
from helpers.dataset_classes.dataset import DatasetBySplit


class Experiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            print(f"{arg}: {value_arg}")
            self.__setattr__(arg, value_arg)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=self.seed)

        # parameters
        self.metric_type = self.dataset.get_metric_type()
        self.decimal = self.dataset.num_after_decimal()
        self.task_loss = self.metric_type.get_task_loss()

        # asserts
        self.dataset.asserts(args)

    def run(self) -> Tuple[Tensor, Tensor]:
        dataset = self.dataset.load(seed=self.seed, pos_enc=self.pos_enc)
        if self.metric_type.is_multilabel():
            dataset.data.y = dataset.data.y.to(dtype=torch.float)

        folds = self.dataset.get_folds(fold=self.fold)

        # locally used parameters
        out_dim = self.metric_type.get_out_dim(dataset=dataset)
        gin_mlp_func = self.dataset.gin_mlp_func()
        env_act_type = self.dataset.env_activation_type()

        # named tuples
        gumbel_args = GumbelArgs(learn_temp=self.learn_temp, temp_model_type=self.temp_model_type, tau0=self.tau0,
                                 temp=self.temp, gin_mlp_func=gin_mlp_func)
        env_args = \
            EnvArgs(model_type=self.env_model_type, num_layers=self.env_num_layers, env_dim=self.env_dim,
                    layer_norm=self.layer_norm, skip=self.skip, batch_norm=self.batch_norm, dropout=self.dropout,
                    act_type=env_act_type, metric_type=self.metric_type, in_dim=dataset[0].x.shape[1], out_dim=out_dim,
                    gin_mlp_func=gin_mlp_func, dec_num_layers=self.dec_num_layers, pos_enc=self.pos_enc,
                    dataset_encoders=self.dataset.get_dataset_encoders())
        action_args = \
            ActionNetArgs(model_type=self.act_model_type, num_layers=self.act_num_layers,
                          hidden_dim=self.act_dim, dropout=self.dropout, act_type=ActivationType.RELU,
                          env_dim=self.env_dim, gin_mlp_func=gin_mlp_func)

        # folds
        metrics_list = []
        edge_ratios_list = []
        for num_fold in folds:
            set_seed(seed=self.seed)
            dataset_by_split = self.dataset.select_fold_and_split(num_fold=num_fold, dataset=dataset)
            best_losses_n_metrics, edge_ratios =\
                self.single_fold(dataset_by_split=dataset_by_split, gumbel_args=gumbel_args, env_args=env_args,
                                 action_args=action_args, num_fold=num_fold)

            # print final
            print_str = f'Fold {num_fold}/{len(folds)}'
            for name in best_losses_n_metrics._fields:
                print_str += f",{name}={round(getattr(best_losses_n_metrics, name), self.decimal)}"
            print(print_str)
            print()
            metrics_list.append(best_losses_n_metrics.get_fold_metrics())

            if edge_ratios is not None:
                edge_ratios_list.append(edge_ratios)

        metrics_matrix = torch.stack(metrics_list, dim=0)  # (F, 3)
        metrics_mean = torch.mean(metrics_matrix, dim=0).tolist()  # (3,)
        if len(edge_ratios_list) > 0:
            edge_ratios = torch.mean(torch.stack(edge_ratios_list, dim=0), dim=0)
        else:
            edge_ratios = None

        # prints
        print(f'Final Rewired train={round(metrics_mean[0], self.decimal)},'
              f'val={round(metrics_mean[1], self.decimal)},'
              f'test={round(metrics_mean[2], self.decimal)}')
        if len(folds) > 1:
            metrics_std = torch.std(metrics_matrix, dim=0).tolist()  # (3,)
            print(f'Final Rewired train={round(metrics_mean[0], self.decimal)}+-{round(metrics_std[0], self.decimal)},'
                  f'val={round(metrics_mean[1], self.decimal)}+-{round(metrics_std[1], self.decimal)},'
                  f'test={round(metrics_mean[2], self.decimal)}+-{round(metrics_std[2], self.decimal)}')
    
        return metrics_mean, edge_ratios
            
    def single_fold(self, dataset_by_split: DatasetBySplit, gumbel_args: GumbelArgs, env_args: EnvArgs,
                    action_args: ActionNetArgs, num_fold: int) -> Tuple[LossesAndMetrics, OptTensor]:
        model = CoGNN(gumbel_args=gumbel_args, env_args=env_args, action_args=action_args,
                      pool=self.pool).to(device=self.device)

        optimizer = self.dataset.optimizer(model=model, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self.dataset.scheduler(optimizer=optimizer, step_size=self.step_size, gamma=self.gamma,
                                           num_warmup_epochs=self.num_warmup_epochs, max_epochs=self.max_epochs)

        with tqdm.tqdm(total=self.max_epochs, file=sys.stdout) as pbar:
            best_losses_n_metrics, edge_ratios =\
                self.train_and_test(dataset_by_split=dataset_by_split, model=model, optimizer=optimizer,
                                    scheduler=scheduler, pbar=pbar, num_fold=num_fold)
        return best_losses_n_metrics, edge_ratios

    def train_and_test(self, dataset_by_split: DatasetBySplit, model, optimizer, scheduler, pbar, num_fold: int)\
            -> Tuple[LossesAndMetrics, OptTensor]:
        train_loader = DataLoader(dataset_by_split.train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_by_split.val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset_by_split.test, batch_size=self.batch_size, shuffle=True)

        best_losses_n_metrics = self.metric_type.get_worst_losses_n_metrics()
        for epoch in range(self.max_epochs):
            self.train(train_loader=train_loader, model=model, optimizer=optimizer)
            train_loss, train_metric, _ =\
                self.test(loader=train_loader, model=model, split_mask_name='train_mask', calc_edge_ratio=False)
            if self.dataset.is_expressivity():
                val_loss, val_metric = train_loss, train_metric
                test_loss, test_metric = train_loss, train_metric
            else:
                val_loss, val_metric, _ =\
                    self.test(loader=val_loader, model=model, split_mask_name='val_mask', calc_edge_ratio=False)
                test_loss, test_metric, _ =\
                    self.test(loader=test_loader, model=model, split_mask_name='test_mask', calc_edge_ratio=False)

            losses_n_metrics = \
                LossesAndMetrics(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                                 train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)
            if scheduler is not None:
                scheduler.step(losses_n_metrics.val_metric)

            # best metrics
            if self.metric_type.src_better_than_other(src=losses_n_metrics.val_metric,
                                                      other=best_losses_n_metrics.val_metric):
                best_losses_n_metrics = losses_n_metrics

            # prints
            log_str = f'Split: {num_fold}, epoch: {epoch}'
            for name in losses_n_metrics._fields:
                log_str += f",{name}={round(getattr(losses_n_metrics, name), self.decimal)}"
            log_str += f"({round(best_losses_n_metrics.test_metric, self.decimal)})"
            pbar.set_description(log_str)
            pbar.update(n=1)

        edge_ratios = None
        if self.dataset.not_synthetic():
            _, _, edge_ratios =\
                self.test(loader=test_loader, model=model, split_mask_name='test_mask', calc_edge_ratio=True)

        return best_losses_n_metrics, edge_ratios

    def train(self, train_loader, model, optimizer):
        model.train()

        for data in train_loader:
            if self.batch_norm and (data.x.shape[0] == 1 or data.num_graphs == 1):
                continue
            optimizer.zero_grad()
            node_mask = self.dataset.get_split_mask(data=data, batch_size=data.num_graphs,
                                                    split_mask_name='train_mask').to(self.device)
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores, _ =\
                model(data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device),
                      batch=data.batch.to(device=self.device), edge_attr=edge_attr, edge_ratio_node_mask=None,
                      pestat=self.pos_enc.get_pe(data=data, device=self.device))
            train_loss = self.task_loss(scores[node_mask], data.y.to(device=self.device)[node_mask])

            # backward
            train_loss.backward()
            if self.dataset.clip_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    def test(self, loader, model, split_mask_name: str, calc_edge_ratio: bool)\
            -> Tuple[float, Any, Tensor]:
        model.eval()

        total_loss, total_metric, total_edge_ratios = 0, 0, 0
        total_scores = np.empty(shape=(0, model.env_args.out_dim))
        total_y = None
        for data in loader:
            if self.batch_norm and (data.x.shape[0] == 1 or data.num_graphs == 1):
                continue
            node_mask = self.dataset.get_split_mask(data=data, batch_size=data.num_graphs,
                                                    split_mask_name=split_mask_name).to(device=self.device)
            if calc_edge_ratio:
                edge_ratio_node_mask =\
                    self.dataset.get_edge_ratio_node_mask(data=data, split_mask_name=split_mask_name).to(device=self.device)
            else:
                edge_ratio_node_mask = None
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores, edge_ratios =\
                model(data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device),
                      edge_attr=edge_attr, batch=data.batch.to(device=self.device),
                      edge_ratio_node_mask=edge_ratio_node_mask,
                      pestat=self.pos_enc.get_pe(data=data, device=self.device))
            
            eval_loss = self.task_loss(scores, data.y.to(device=self.device))

            # analytics
            total_scores = np.concatenate((total_scores, scores[node_mask].detach().cpu().numpy()))
            if total_y is None:
                total_y = data.y.to(device=self.device)[node_mask].detach().cpu().numpy()
            else:
                total_y = np.concatenate((total_y, data.y.to(device=self.device)[node_mask].detach().cpu().numpy()))

            total_loss += eval_loss.item() * data.num_graphs
            total_edge_ratios += edge_ratios * data.num_graphs

        metric = self.metric_type.apply_metric(scores=total_scores, target=total_y)

        loss = total_loss / len(loader.dataset)
        edge_ratios = total_edge_ratios / len(loader.dataset)
        return loss, metric, edge_ratios
