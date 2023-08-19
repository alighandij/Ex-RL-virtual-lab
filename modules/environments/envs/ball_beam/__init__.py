import numpy as np
from .create import create
from .encoders import ENCODERS

BALL_BEAM = {
    "Ball Beam": {
        "create": create,
        "state_shape": 3,
        "parameters": {
            "beam_length": {
                "options": np.arange(0.25, 5.25, 0.25),
                "default": 1
            }
        },
        "agent_configs": {
            "target": 50,
            "discrete": 8,
            "break_on_solve": False,
        },
        "encoders": ENCODERS
    }
}
