import numpy as np


def q1(x: np.ndarray) -> float:
    return x.quantile(0.25)


def q3(x: np.ndarray) -> float:
    return x.quantile(0.75)
