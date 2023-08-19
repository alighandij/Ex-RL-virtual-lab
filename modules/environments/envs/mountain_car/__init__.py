import numpy as np
from .create import create
from .encoders import ENCODERS

MOUNTAIN_CAR = {
    "Mountain Car": {
        "create": create,
        "state_shape": 2,
        "parameters": {
            "force": {
                "options": np.linspace(0.001 / 2, 0.001 * 2, 4),
                "default": 1e-3,
            },
            "gravity": {
                "options": np.linspace(0.0025 / 2, 0.0025 * 2, 4),
                "default": 0.0025,
            },
            "max_speed": {
                "options": np.linspace(0.07 / 2, 0.07 * 2, 4),
                "default": 0.07,
            },
            "angle_perturbation": {
                "options": list(range(-10, 11)),
                "default": 0,
            }
        },
        "agent_configs": {
            "discrete": 32,
            "count": 20,
            "target": -120
        },
        "encoders": ENCODERS
    }
}
