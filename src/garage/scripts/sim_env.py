import akro
import gym
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType

class PointEnv(Environment):

    # ...

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x**2 + y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return next_observation, reward, done, None
    @property
    def observation_space(self):
        return akro.Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return akro.Box(low=-0.1, high=0.1, shape=(2,))
    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation
    def render(self):
        print ('current state:', self._state)

