import os, random, numpy as np
import tensorflow as tf

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dirs():
    for p in ["artifacts","artifacts/models","artifacts/figures","artifacts/reports","data/processed"]:
        os.makedirs(p, exist_ok=True)
