import numpy as np
from .create import create
from .encoders import ENCODERS

PENDULUM = {
    "Pendulum": {
        "create": create,
        "state_shape": 3,
        "parameters": {
            "gravity": {
                "options": np.linspace(10.0 / 2, 10.0 * 2, 4),
                "default": 10.0,
            },
            "max_speed": {
                "options": np.linspace(8 / 2, 8 * 2, 4),
                "default": 8,
            },
            "max_torque": {
                "options": np.linspace(2.0 / 2, 2.0 * 2, 4),
                "default": 2.0,
            },
            "mass": {
                "options": np.linspace(1.0 / 2, 1.0 * 2, 4),
                "default": 1.0,
            },
            "length": {
                "options": np.linspace(1.0 / 2, 1.0 * 2, 4),
                "default": 1.0,
            },
        },
        "agent_configs": {
            "discrete": 8,
            "count": 20,
            "target": -5
        },
        "encoders": ENCODERS
    }
}
