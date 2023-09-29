from torch_geometric.graphgym.loader import index2mask, set_dataset_attr


TASK = 'graph'


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices

    Raises:
        ValueError: If any pair of splits has intersecting indices
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )

    task_level = TASK
    if task_level == 'node':
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for split_name, split_index in zip(split_names, splits):
            mask = index2mask(split_index, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_name, mask, len(mask))

    elif task_level == 'graph':
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(dataset, split_name, split_index, len(split_index))

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")
