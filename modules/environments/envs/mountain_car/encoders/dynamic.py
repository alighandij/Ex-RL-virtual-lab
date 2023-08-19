import numpy as np
from exrl.custom_mountaincar import AngleModifiedDiscreteMountainCar

def phase_mapping(env, state: np.ndarray, next_state: np.ndarray) -> int:
    def h(y): return env._height(y)
    def s(x): return np.sign(x)
    dx = next_state[0] - state[0]
    dy = h(next_state[0]) - h(state[0])
    key = (s(dx), s(dy))

    return {
        (-1, +1): 0,
        (+1, -1): 1,
        (+1, +1): 2,
        (-1, -1): 3,
    }.get(key, 4)
    
def encode_trajectory(slope: int, trajectories: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        slope (int): slope AKA Angle Perturbation of givin trajectories 
        trajectories (np.ndarray): ([S0, S1], [S1, S2], ...., [Sn-1, Sn])
    Returns:
        np.ndarray: Encoded Trajectories
    """
    env = AngleModifiedDiscreteMountainCar(angle_perturbation=slope)
    lst = [phase_mapping(env, *state) for state in trajectories]
    return lst

def dynamic_encoding(samples: dict, pool):
    sequences = []
    for agent_id, values in samples.items():
        for trajectories in values:
            slope = pool.get_agent_property(agent_id, "angle_perturbation")
            encoded = encode_trajectory(slope, trajectories)
            sequences.append(encoded)
    return sequences