import akro
import gym
import numpy as np
import math 

from garage import Environment, EnvSpec, EnvStep, StepType

class TestPointEnv(Environment):

    # ...
    def __init__(self, max_episode_length=math.inf): 
    
        self._spec = EnvSpec(action_space=self.action_space,
                            observation_space=self.observation_space,
                            max_episode_length=max_episode_length)


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

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]
    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation
    def render(self):
        print ('current state:', self._state)

    def close(self):
        """Close the env."""

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('ascii'))
        
    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks