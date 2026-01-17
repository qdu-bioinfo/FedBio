def load_CRC_data(idx, output_dir, k_fold, groups, is_train=True,):
    """
    Load pre-generated train or test data for a specific study
    Args:
        idx: Study index to load
        output_dir: Directory where splits are saved
        is_train: Whether to load train or test data
    Returns:
        List of (feature, label) tuples as torch tensors
    """
    import os
    import pickle
    import torch

    file_path = os.path.join(output_dir, f'{"".join(groups)}_fold{k_fold}_study{idx}.pkl')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved data found for study {idx} in {output_dir}")

    # Load the data
    with open(file_path, 'rb') as f:
        study_data = pickle.load(f)

    if is_train:
        X_gene = torch.Tensor(study_data['X_mic_train']).type(torch.float32)
        X_meta = study_data['X_meta_train']
        y = torch.Tensor(study_data['y_train']).type(torch.int64)
    else:
        X_gene = torch.Tensor(study_data['X_mic_test']).type(torch.float32)
        X_meta = study_data['X_meta_test']
        y = torch.Tensor(study_data['y_test']).type(torch.int64)

    return [(x_gene, x_meta, label) for x_gene, x_meta, label in zip(X_gene, X_meta, y)]