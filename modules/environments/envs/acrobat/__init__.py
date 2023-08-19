from .create import create
from .encoders import ENCODERS

ACROBAT = {
    "Acrobat": {
        "create": create,
        "state_shape": 6,
        "parameters": {},
        "agent_configs": {
            "discrete": 8,
            "count": 20,
            "target": -200
        },
        "encoders": ENCODERS
    }
}
