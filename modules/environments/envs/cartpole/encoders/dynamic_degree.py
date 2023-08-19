import numpy as np


def phase_mapping_degree(state: np.ndarray, next_state: np.ndarray) -> int:
    def s(x): return np.sign(x)

    dx = next_state[2] - state[2]
    dy = state[2]
    key = (s(dx), s(dy))

    return {
        (-1, -1): 0,
        (+1, -1): 1,
        (+1,  0): 1,
        (0, -1): 1,
        (-1,  0): 3,
        (-1, +1): 3,
        (0, +1): 3,
        (+1, +1): 2,
        (0,  0): 1, #!ٌWARNING ITS 4 ACTUALLY
    }.get(key, None)


def dynamic_encoding_degree(samples: dict, pool):
    sequences = []
    for agent_id, values in samples.items():
        for trajectories in values:
            encoded = [phase_mapping_degree(*states) for states in trajectories]
            sequences.append(encoded)
    return sequences
