import torch
import random
import numpy as np
import os

# Set configuration
class cfg():
    # fixed
    data_dir = '/home/dataset/chexpert/CheXpert-v1.0-small'
    out_size = 3 # ap/pa/lateral
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tune
    seed = 42
    num_classes = 5
    num_workers = 4
    batch_size = 32
    epochs = 5
    scheduler = 'CosineAnnealingLR'
    learning_rate = 1e-4


# Fix random seeds
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True