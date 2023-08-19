import numpy as np
from .create import create
from .encoders import ENCODERS

CART_POLE = {
    "Cart Pole V1": {
        "create": create,
        "state_shape": 4,
        "parameters": {
            "gravity": {
                "options": np.linspace(9.8 / 2, 9.8 * 2, 16),
                "default": 9.8,
            },
            "masscart": {
                "options": np.linspace(1.0 / 2, 1.0 * 2, 16),
                "default": 1.0,
            },
            "masspole": {
                "options": np.linspace(0.1 / 2, 0.1 * 2, 16),
                "default": 0.1,
            },
            "force_mag": {
                "options": np.linspace(10 / 2, 10 * 2, 16),
                "default": 10.0,
            },
        },
        "agent_configs": {
            "discrete": 8,
            "count": 20,
            "target": 200
        },
        "encoders": ENCODERS
    }
}
