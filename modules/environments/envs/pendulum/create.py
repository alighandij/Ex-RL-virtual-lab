from .pendulum import PendulumEnv
from exrl.gym_wrapper import GymWrapper

MAX_STEP = 200


def is_goal_reached(env: GymWrapper):
    return False


def create(**kwargs):
    env = PendulumEnv(**kwargs)
    return GymWrapper(env, MAX_STEP, lambda x: x.episode_step < 200)
