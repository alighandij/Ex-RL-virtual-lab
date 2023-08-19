from .envs.acrobat import ACROBAT
from .envs.pendulum import PENDULUM
from .envs.cartpole import CART_POLE
from .envs.ball_beam import BALL_BEAM
from .envs.mountain_car import MOUNTAIN_CAR
ENVIRONMENTS = {
    **ACROBAT,
    **PENDULUM,
    **CART_POLE,
    **BALL_BEAM,
    **MOUNTAIN_CAR,
}

AGENT = "agent_configs"
DEFAULT = "default"
OPTIONS = "options"
ENCODERS = "encoders"
PARAMETERS = "parameters"


class EnvSelector:

    @staticmethod
    def get_env_names(sorted: bool = True) -> list[str]:
        names = list(ENVIRONMENTS.keys())
        if sorted:
            names.sort()
        return names

    @staticmethod
    def get_env_params(env_name: str) -> tuple[str]:
        return tuple(ENVIRONMENTS[env_name][PARAMETERS].keys())

    @staticmethod
    def get_env_param(env_name: str, parameter: str):
        options = ENVIRONMENTS[env_name][PARAMETERS][parameter][OPTIONS]
        default = ENVIRONMENTS[env_name][PARAMETERS][parameter][DEFAULT]
        return options, default

    @staticmethod
    def get_env_maker(env_name: str):
        return ENVIRONMENTS[env_name]["create"]

    @staticmethod
    def get_state_shape(env_name: str):
        return ENVIRONMENTS[env_name]["state_shape"]

    @staticmethod
    def _get_agent_param(env_name: str, param: str, default):
        agent = ENVIRONMENTS[env_name].get(AGENT, {})
        return agent.get(param, default)

    @staticmethod
    def get_agent_gamma(env_name: str):
        return EnvSelector._get_agent_param(env_name, "gamma", 0.99)

    @staticmethod
    def get_agent_episodes(env_name: str):
        return EnvSelector._get_agent_param(env_name, "episodes", 20_000)

    @staticmethod
    def get_agent_lr(env_name: str):
        return EnvSelector._get_agent_param(env_name, "lr", 1.0)

    @staticmethod
    def get_agent_lr_min(env_name: str):
        return EnvSelector._get_agent_param(env_name, "lr_min", 0.01)

    @staticmethod
    def get_agent_lr_decay(env_name: str):
        return EnvSelector._get_agent_param(env_name, "lr_decay", 0.999)

    @staticmethod
    def get_agent_epsilon(env_name: str):
        return EnvSelector._get_agent_param(env_name, "epsilon", 1.0)

    @staticmethod
    def get_agent_epsilon_decay(env_name: str):
        return EnvSelector._get_agent_param(env_name, "epsilon_decay", 0.99)

    @staticmethod
    def get_agent_epsilon_min(env_name: str):
        return EnvSelector._get_agent_param(env_name, "epsilon_min", 1e-3)

    @staticmethod
    def get_agent_discrete(env_name: str):
        return EnvSelector._get_agent_param(env_name, "discrete", 32)

    @staticmethod
    def get_agent_count(env_name: str):
        return EnvSelector._get_agent_param(env_name, "count", 20)

    @staticmethod
    def get_agent_target(env_name: str):
        return EnvSelector._get_agent_param(env_name, "target", 100_000)

    @staticmethod
    def get_agent_break_on_solve(env_name: str):
        return EnvSelector._get_agent_param(env_name, "break_on_solve", False)

    @staticmethod
    def get_agent_reward_shaper(env_name: str):
        return EnvSelector._get_agent_param(env_name, "reward_shaper", None)

    @staticmethod
    def get_encoders_names(env_name: str):
        return list(ENVIRONMENTS[env_name][ENCODERS].keys())

    @staticmethod
    def get_encoder(env_name: str, encoder_name: str):
        return ENVIRONMENTS[env_name][ENCODERS][encoder_name]["encoder"]

    @staticmethod
    def get_phase_map(env_name: str, encoder_name: str):
        return ENVIRONMENTS[env_name][ENCODERS][encoder_name]["phase_map"]

    @staticmethod
    def get_encode_count(env_name: str, encoder_name: str):
        return ENVIRONMENTS[env_name][ENCODERS][encoder_name]["encode_count"]
