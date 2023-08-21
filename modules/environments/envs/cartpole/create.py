from .cart_pole import CartPoleEnv
from exrl.gym_wrapper import GymWrapper


def create(**kwargs):
    return CartPoleEnv(**kwargs)
