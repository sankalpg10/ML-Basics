

import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    max_val = np.max(scores)

    scores = scores - max_val

    log_softmax_arr = scores - np.log(np.sum(np.exp(scores)))

    return log_softmax_arr


print(log_softmax([1, 2, 3]))