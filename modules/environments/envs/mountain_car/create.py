import numpy as np
from gym.vector.utils import spaces
from exrl.custom_mountaincar import AngleModifiedDiscreteMountainCar

def create(**kwargs):
    force = kwargs.get("force")
    gravity = kwargs.get("gravity")
    max_speed = kwargs.get("max_speed")
    angle_perturbation = kwargs.get("angle_perturbation")

    env = AngleModifiedDiscreteMountainCar(
        angle_perturbation=angle_perturbation,
        is_continuous=True
    )

    env.force = force
    env.gravity = gravity
    env.max_speed = max_speed
    env.min_speed = -max_speed

    env.low = np.array([env.min_position, -max_speed], dtype=np.float32)
    env.high = np.array([env.max_position, max_speed], dtype=np.float32)

    env.observation_space = spaces.Box(env.low, env.high, dtype=np.float32)

    return env