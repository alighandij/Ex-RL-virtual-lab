import math
import numpy as np
from abc import ABC, abstractmethod
from gym.vector.utils import spaces
from gym.error import DependencyNotInstalled
from numpy.random import RandomState

class Environment(ABC):
    """
    This class provides an abstraction over hand-made and gym-based environments.
    """
    def __init__(self, max_steps=-1):
        self.max_steps = None
        self.set_max_steps(max_steps)

    @abstractmethod
    def get_n_actions(self):
        """
        returns the number of actions in case of discrete action space.
        :return: int or None. number of actions or None if action space is continuous.
        """
        pass

    @abstractmethod
    def get_action_shape(self):
        """
        returns the action vector shape in case of continuous action space.
        :return: np.array or None. action vector shape or None if action space is discrete.
        """
        pass

    @abstractmethod
    def get_obs_shape(self):
        """
        In continuous case, returns the shape of the observation vector.
        In discrete case, returns the tabular dimensions of the state space.
        :return: np.array. state vector shape
        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets the environment to the initial state. It is supposed to have the same functionality as
            gym.env.reset()
        :return: np.array or None. initial state of the episode or None if environment is a bandit env.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        steps in the environment with the action given. It is supposed to have the same functionality as
            gym.env.step() but it has max_steps specified by self.max_steps.
        :param action: int or np.array. action to take in the current state.
        :return:  (reward, next_state, done)
        """
        pass

    def set_max_steps(self, max_steps=-1):
        """
        set the maximum number of steps in an episode. if -1, episodes will elongate as long as possible.
        :param max_steps: int. maximum number of steps. should be either positive or -1
        :return: None
        """
        self.max_steps = max_steps

    def get_max_steps(self):
        """
        return the current maximum number of steps set for the environment.
        :return: int
        """
        return self.max_steps


class AngleModifiedDiscreteMountainCar(Environment):
    """
    This class is a Mountain Car environment with 2 adjustments:
        1. It is discretized. Although you can Change the granularity of the discretization,
            it is not meant to be modified for the purpose it was created and is mostly meant to be used
            with the default parameter.
        2. Right-side hill's angle is adjustable. The right hill's angle is defined as the "Angle of the line which
            connects the lowest point of the road to the flag position, and the horizontal axis of the environment".
            By adjusting this value one can make the right side of the hill to be steeper or more flattened.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, granularity=(32, 32), angle_perturbation=0, is_continuous=False):
        """
        :param granularity: a 2-element list or tuple indicating the number of intervals by which to divide each
                            dimension of state space. default value is (32,32) which means that state space is
                            going to be discretized to a 32x32 matrix.
        :param angle_perturbation: the amount (in degrees) of perturbation of the right-side hill. can be either
                            positive or negative. 0 means no perturbation while negative numbers result in more
                             flattened hill and positive numbers lead to steeper hill.
        """
        super().__init__()
        self.granularity = granularity
        self.angle_perturbation = angle_perturbation

        self.max_steps = 200
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.min_speed = -self.max_speed
        self.goal_position = 0.5
        self.goal_velocity = 0

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.state = None
        self.screen = None
        self.scr_width = 600
        self.scr_height = 600
        self.car_width = 40
        self.car_height = 20
        self.clock = None
        self.is_open = True
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.step_counter = 0

        # constants for calculation
        self.bottom_height = AngleModifiedDiscreteMountainCar._default_height(-math.pi/6)
        self.old_goal_height = AngleModifiedDiscreteMountainCar._default_height(0.5)
        self.default_angle = math.atan((self.old_goal_height - self.bottom_height)/1.1)
        self.angle_perturbation_radian = self.default_angle + self.angle_perturbation * 2 * math.pi / 360
        self.new_goal_height = self.bottom_height + math.tan(self.angle_perturbation_radian) * 1.1

        self.is_continuous = is_continuous

        self.rng = np.random.RandomState(4)

    def discretize_state(self, state):
        x, v = state
        eps = 1e-6
        x_idx = int((x-self.min_position)/(self.max_position - self.min_position) * (self.granularity[0]-eps))
        v_idx = int((v-self.min_speed) / (self.max_speed - self.min_speed) * (self.granularity[1]-eps))

        return int(x_idx), int(v_idx)

    def _discretize_current_state(self):
        return self.discretize_state(self.state)
        

    def update_position(self):
        pos, vel = self.state
        pos += vel

        pos = np.clip(pos, self.min_position, self.max_position)

        if pos == self.max_position and vel > 0:
            vel = 0
        if pos == self.min_position and vel < 0:
            vel = 0

        self.state = pos, vel

    def update_velocity(self, action):
        pos, vel = self.state

        update = action - 1
        coeff = int(pos > -math.pi/6)*(self.new_goal_height/self.old_goal_height - 1) + 1
        vel += np.clip(update * self.force + coeff*math.cos(3 * pos) * (-self.gravity), -self.max_speed, self.max_speed)

        self.state = pos, vel

    def is_goal_reached(self):
        return self.state[0] > self.goal_position

    def is_max_steps_reached(self):
        return self.step_counter >= self.max_steps

    def check_termination(self):
        if self.is_goal_reached():
            return True
        elif self.is_max_steps_reached():
            return True
        else:
            return False

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        self.step_counter += 1

        self.update_velocity(action)
        self.update_position()

        done = self.check_termination()
        reward = -1.0
        
        if self.is_continuous:
            return self.state, reward, done, {"info": None}

        return self._discretize_current_state(), reward, done

    def reset(self, fix_start_point=False, starting_point=None):
        if fix_start_point and starting_point is not None:
            self.state = np.array(starting_point)
        else:
            self.state = np.array([self.rng.uniform(low=-0.6, high=-0.4), 0])

        self.step_counter = 0
        
        if self.is_continuous:
            return self.state
        
        return self._discretize_current_state()

    @staticmethod
    def _default_height(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def _height(self, xs):
        C = self.new_goal_height/self.old_goal_height
        ind = (np.array(xs) > -math.pi/6)
        def_heights = AngleModifiedDiscreteMountainCar._default_height(xs)
        return (ind*(C-1) + 1)*def_heights + ind*(1-C)*0.1

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.scr_width, self.scr_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        pos, vel = self.state
        surf = self._render_dim(pos, self._height)

        self.screen.blit(surf, (0, 0))

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.is_open

    def _render_dim(self, pos, height_fn):
        import pygame
        from pygame import gfxdraw

        world_width = self.max_position - self.min_position
        scale = self.scr_width / world_width

        surf = pygame.Surface((self.scr_width, self.scr_height))
        surf.fill((255, 255, 255))

        # Draw the track curve
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = height_fn(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))

        # Draw the car
        clearance = 10
        l, r, t, b = -self.car_width / 2, self.car_width / 2, self.car_height, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + height_fn(pos) * scale,
                )
            )

        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

        # Draw the wheels
        for c in [(self.car_width / 4, 0), (-self.car_width / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + height_fn(pos) * scale),
            )

            gfxdraw.aacircle(
                surf, wheel[0], wheel[1], int(self.car_height / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                surf, wheel[0], wheel[1], int(self.car_height / 2.5), (128, 128, 128)
            )

        # Draw the flag
        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(height_fn(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        # Make it human friendly
        surf = pygame.transform.flip(surf, False, True)

        return surf

    @staticmethod
    def render_episode(episode_states, episode_env):
        _ = episode_env.reset()
        for s in episode_states:
            episode_env.state = s
            episode_env.render()
        return

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.is_open = False

    def get_n_actions(self):
        return 3

    def get_action_shape(self):
        return ()

    def get_obs_shape(self):
        return self.granularity