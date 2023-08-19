import numpy as np
from exrl.custom_mountaincar import AngleModifiedDiscreteMountainCar

def phase_mapping(env, state: np.ndarray, next_state: np.ndarray) -> int:
    return
    
def encode_trajectory(slope: int, trajectories: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        slope (int): slope AKA Angle Perturbation of givin trajectories 
        trajectories (np.ndarray): ([S0, S1], [S1, S2], ...., [Sn-1, Sn])
    Returns:
        np.ndarray: Encoded Trajectories
    """
    # lst = [phase_mapping(env, *state) for state in trajectories]
    return

def dynamic_encoding(samples: dict, pool):
    sequences = []
    for agent_id, values in samples.items():
        for trajectories in values:
            continue
            sequences.append(encoded)
    return sequences