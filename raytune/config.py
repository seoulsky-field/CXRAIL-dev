import torch
import random
import numpy as np
import os

# Set configuration
class cfg():
    num_classes: 3
    root: "/home/dataset/chexpert"
    train_csv: "/home/dataset/chexpert/CheXpert-v1.0-pad224/train.csv"
    valid_csv: "/home/dataset/chexpert/CheXpert-v1.0-pad224/valid.csv"
    ckpt_name: "best_saved.pth"
    num_workers: 1
    batch_size: 16
    epochs: 5
    scheduler: "CosineAnnealingLR"
    learning_rate: 1e-4
    weight_decay: 1e-4



# Fix random seeds
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True