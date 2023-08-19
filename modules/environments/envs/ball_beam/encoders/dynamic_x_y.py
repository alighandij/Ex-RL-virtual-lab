import math
import numpy as np


def phase_mapping_x_y(state: np.ndarray, next_state: np.ndarray) -> int:
    def s(x): return np.sign(x)
    def sin(y): return math.sin(y)
    def cos(y): return math.cos(y)

    dx = sin(next_state[0]) - sin(state[0])
    dy = cos(next_state[0]) - cos(state[0])
    key = (s(dx), s(dy))

    return {
        (-1, +1): 0,
        (+1, -1): 1,
        (+1, +1): 2,
        (-1, -1): 3,
    }.get(key, 4)



def dynamic_encoding_x_y(samples: dict, pool):
    sequences = []
    for agent_id, values in samples.items():
        for trajectories in values:
            encoded = [phase_mapping_x_y(*states) for states in trajectories]
            sequences.append(encoded)
    return sequences
