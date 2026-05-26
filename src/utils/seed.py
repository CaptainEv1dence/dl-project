def set_seed(seed: int) -> None:
    """Seed Python random, numpy, torch, and CUDA (if available) for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
