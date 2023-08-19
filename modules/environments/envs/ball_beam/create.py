import gym
import ballbeam_gym

def create(**kwargs):
    beam_length = kwargs.get("beam_length")
    config = {
        'timestep': 0.05,
        'max_angle': 0.2,
        'beam_length': beam_length,
        'action_mode': 'discrete',
        'init_velocity': 0.0,
    }
    env = gym.make("BallBeamBalance-v0", **config)
    return env
