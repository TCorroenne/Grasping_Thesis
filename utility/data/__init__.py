def get_custom_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'custom':
        from .cornell_data import CustomDataset
        return CustomDataset
    # elif dataset_name == 'jacquard':
    #     from .jacquard_data import JacquardDataset
    #     return JacquardDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))