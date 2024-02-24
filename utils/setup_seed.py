import os
import torch
import numpy as np
import random


def setup_seed(seed=3407):
    """
    Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
