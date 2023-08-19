import os
import numpy as np
from glob import glob

def save_np_arr(arr, name):
    np.save(name, arr)


def load_np_arr(name):
    with open(name, "rb") as f:
        return np.load(f)


def load_datasets():
    datasets = glob("./Datasets/*")
    datasets = map(load_np_arr, datasets)
    datasets = list(datasets)
    datasets = np.concatenate(datasets)
    return datasets



def make_dir(name: str):
    try:
        os.mkdir(name)
        return True
    except:
        return False