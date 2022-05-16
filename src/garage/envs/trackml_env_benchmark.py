"""Simple 2D environment containing a point and a goal location."""
from concurrent.futures import process
import math

import akro
from charset_normalizer import detect
import numpy as np
import circle_fit as cf
import pandas as pd 
from garage import Environment, EnvSpec, EnvStep, StepType
import random 
from gym.spaces import Box
import csv
from sklearn.metrics import precision_recall_curve 
import trackml.dataset
import json
import yaml



prefix = '/home/lhv14/exatrkx/Tracking-ML-Exa.TrkX/alldata/train_1_processed1000/processed_event00000'
f = open("garage_outputs.csv", "w")
writer = csv.writer(f)
writer.writerow(["filenumber", "particle_id", "mc_z", "mc_r", "pred_z", "pred_r", "action_z", "action_r"])

class TrackMLEnvBenchmark(Environment):
    """A simple 2D point environment.

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
        max_episode_length (int): The maximum steps allowed for an episode.

    """

    def __init__(self,
                # goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 done_bonus=0.,
                 never_done=False,
                 max_episode_length=500):
        #goal = np.array(goal, dtype=np.float32)
        #self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size
        self._total_step_cnt = 0 
        self.new_count = 0 
        self.done_visual = False 
        self.file_counter = 0 
        #self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
        #self.event = pd.read_hdf('/media/lucas/MicroSD/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
        #self.event['z'] = np.abs(self.event['z'])
        self.average_reward = 0 
        self.hit_buffer = []
        self.dz_buffer = []
        self.dr_buffer = []


       # assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        #self._point = np.zeros_like(self._goal)
        #self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=np.array([-300, 0]), high=np.array([300, 100]), dtype=np.float64)
        self._action_space = akro.Box(low=np.array([-50, 0]),
                                      high=np.array([50,25]),
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

        self.record_z = [] 
        self.record_r = []
        self.record_pid = []
        self.record_event_counter = [] 
        self.record_reward = [] 
        self.record_a0 = [] 
        self.record_a1 = [] 
        self.record_filenumber = [] 

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditisoned or MTRL.)

        """
        
        if self._total_step_cnt%100 ==0: 
            self.file_counter += 1 
            #self.event = process_trackml(self.file_counter, pt_min=2)
            
            #self.event['z'] = np.abs(self.event['z'])
            self.event = pd.read_csv(prefix + str(1000+self.file_counter) + '.csv')
            self.event =  self.event[self.event['pt'] > 2]
            self.event = self.event[self.event['nhits'] > 7]
            
            #print("jumping file")
      

        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]

        self.original_pid = random_particle_id
        start_hit = self.particle.iloc[0,:]
        next_hit = self.particle.iloc[1,:]
        dr = next_hit.r - start_hit.r
        dz = next_hit.z - start_hit.z 
        self._point = next_hit[['z', 'r']].values 
        self.dr_buffer = [dr]
        self.dz_buffer = [dz]

        self.num_track_hits = 1

       
        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [start_hit.z],
        'mc_r' : [start_hit.r],
        'pred_z': [start_hit.z],
        'pred_r': [start_hit.r],
        'action_z': [np.nan],
        'action_r': [np.nan]})
        row.to_csv(f, mode='a', header=None, index=None)

        observation=self._point

        
        self.state = [next_hit.z, next_hit.r]
        self.module_id = next_hit.discrete_module_id

        self._step_cnt = 0
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()

        return observation, dict(something=[1,1])

    def step(self, action):
        """Step the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        
        self.new_count += 1 

        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
     
        a_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        predicted_point_z = np.clip(self._point[0] + a_clipped[0] +max(0, self.dz_buffer[-1]), -266, 266)
        predicted_point_r = np.clip(self._point[1] + a_clipped[1] + max(0, self.dr_buffer[-1]), 0, 27)


        predicted_point = [predicted_point_z, predicted_point_r]

        if self._visualize:
            print(self.render('ascii'))

        #mag_dist_prev_hit = np.sqrt(self.state.z-new_hit.z)**2 + (self.state.r-new_hit.r)**2
        self.previous_state = self.state
        self.state = predicted_point

        next_index = self.num_track_hits + 1 
        if next_index > len(self.original_particle) -1: 
            next_index = len(self.original_particle) - 1
        next_hit = self.original_particle.loc[next_index,: ]
       
        distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
        #print(distance)
        reward = -distance
     
        
        self.num_track_hits += 1 
    
        dr = self.state[1] - self.previous_state[1]
        dz = self.state[0] - self.previous_state[0]
       
        self.dr_buffer.append(dr)
        self.dz_buffer.append(dz)
      
        self._step_cnt += 1
        self._total_step_cnt += 1
  

        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [next_hit.z], 
        'mc_r' : [next_hit.r], 
        'pred_z': [predicted_point_z], 
        'pred_r': [predicted_point_r], 
        'action_z': [a[0]], 
        'action_r': [a[1]] })
        row.to_csv(f, mode='a', header=None, index=None)

        if self.num_track_hits > 6: 
        #if a[2] > 0.5:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = predicted_point
        #print(self._point, "\n")
        observation = self._point 
        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)
 
        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None


        self.average_reward = (self.average_reward + reward)/2
        #if self._total_step_cnt%100==0: 
        #    print(self.average_reward)

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=observation, 
                       env_info={
                           #'task': self._task,
                        #'filenumber': self.file_counter, 
                        #'particle_id': self.original_pid, 
                        #'mc_z': next_hit.z, 
                        #'mc_r' : next_hit.r, 
                        #'pred_z': predicted_point_z, 
                        #'pred_r': predicted_point_r, 
                        #'action_z': a[0], 
                        #'action_r': a[1], 
                        #'allowed_next_modules':np.asarray(self.allowed_next_modules)
                        'module_id': self.module_id, 
                           #'success': succ
                       },
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """
        #return f'Point: {self._point}, Goal: {self._goal}'
        return self._point 

    def visualize(self):
        """Creates a visualization of the environment."""
        #self._visualize = True
        #print(self.render('ascii'))
        #visualise(self.state.r, )
        #visualise() 
        #wrap(self.event, r, z, pids, self.original_pid)
        print(self.original_pid)
        #print("i now visualise")

    def my_visualise(self): 
            print("now calling visualise")
            #wrap(self.event)
            #visualise(self.event, self.record_pid, self.record_r, self.record_z)
        
    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        #goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        #tasks = [{'goal': goal} for goal in goals]
        #return tasks
        return 0 


    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        #self._task = task
        #self._goal = task['goal']
        x = 10 

 