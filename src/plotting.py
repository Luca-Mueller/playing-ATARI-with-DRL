import numpy as np
import pickle
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from pathlib import Path
from typing import Tuple, Union


def plot_scores(scores, title: str = None, show: bool = True):
    scores = np.array(scores, dtype=np.float32)
    if title is not None:
        plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores)

    if show:
        plt.show()

    return plt.gcf()
