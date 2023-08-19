from .cart_pole import CartPoleEnv
from exrl.gym_wrapper import GymWrapper

MAX_STEP = 500

def is_goal_reached(env: GymWrapper):
    return env.episode_step >= MAX_STEP

def create(**kwargs):
    env = CartPoleEnv(**kwargs)
    return GymWrapper(env, MAX_STEP, is_goal_reached)
