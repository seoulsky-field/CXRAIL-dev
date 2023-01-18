import torch
import random
import os
import numpy as np


# Seed
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    print(f"SUCCESS {seed} SEED FIXING")
