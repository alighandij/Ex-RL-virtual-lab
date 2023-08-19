from .acrobat import AcrobotEnv
from exrl.gym_wrapper import GymWrapper

MAX_STEP = 500

def is_goal_reached(env: GymWrapper):
    return env.env._terminal()

def create(**kwargs):
    env = AcrobotEnv(**kwargs)
    return GymWrapper(env, MAX_STEP, is_goal_reached)
