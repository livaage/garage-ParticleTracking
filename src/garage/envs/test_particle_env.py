"""Simple 2D environment containing a point and a goal location."""
import math

import akro
import numpy as np
import pandas as pd 
from garage import Environment, EnvSpec, EnvStep, StepType
import random 
from gym.spaces import Box
#from visualise_track import visualise 
from animate_particle import wrap 

r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)

#i = np.where(pids.values.flatten()==-17737)

#my_r = r.values[i]
#my_z = z.values[i]
done_ani = False 

event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')


class ParticleEnv(Environment):
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
                 goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 done_bonus=0.,
                 never_done=False,
                 max_episode_length=math.inf):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size
        self._total_step_cnt = 0 
        self.new_count = 0 
        self.done_visual = False 



        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=np.array([-266, 0]), high=np.array([266, 26]), dtype=np.float64)
        self._action_space = akro.Box(low=-4,
                                      high=20,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

        self.record_z = [] 
        self.record_r = []
        self.record_pid = []
        print("INIIIITIALLIIISED")

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
                goal-conditioned or MTRL.)

        """

        #self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
        self.event = event[event['sim_pt'] > 2]
        #subset by the number of hits 
        nh = self.event.groupby('particle_id').agg('count').iloc[:,0]
        # only pick the pids that has a certain number of hits 
        self.event = self.event[self.event['particle_id'].isin(np.array(nh[nh > 7].index))]

        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]
        self.original_pid = random_particle_id
        # This relies on an ordered df!  
        start_hit = self.particle.iloc[0,:]
        self._point = start_hit[['z', 'r']].values 
        next_hit = self.particle.iloc[1,:]
        self.num_track_hits = 0 
        dist = np.linalg.norm(start_hit[['z', 'r']].values - next_hit[['z', 'r']].values)        
        #print(self._point, dist)
        self.state = start_hit.squeeze(axis=0) 

        first_obs = np.concatenate([self._point, (dist, )])

        self._step_cnt = 0
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()

        return self._point, dict(goal=self._goal)

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

        #print("i am stepping so new count is ", self.new_count)
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        #self._point = np.clip(self._point + a, -266)
        #                      266)
        
        
        if self._visualize:
            print(self.render('ascii'))

        other_hits = self.event[self.event['hit_id']!=self.state.hit_id]
        # it's a big search, converting to list from pandas save an order of magnitude in time,a also just search a small part of the df 
        zlist = other_hits.z.tolist()
        rlist = other_hits.r.tolist() 

        distances = np.sqrt((zlist-self._point[0])**2+(rlist - self._point[1])**2) 
        index = np.argmin(distances)
        
        new_hit = other_hits.iloc[index, ] 
        distance_prev_hit = np.sqrt((self.state.r - new_hit.r)**2 + (self.state.z - new_hit.z)**2)
      
        self.state = new_hit 

        # this is dangerous - relies on ordered df! 
        next_hit = self.original_particle.loc[self.num_track_hits +1,: ]
        #reward given based on how close this new hit was to the next hit in the df 
        distance = np.sqrt((new_hit.z - next_hit.z)**2 + (new_hit.r - next_hit.r)**2)
        reward = -distance
        #if (distance_prev_hit < 1): 
        #    reward -=100

        self.num_track_hits += 1 
        
        self.record_pid.append(self.original_pid)
        self.record_z.append(new_hit.z)
        self.record_r.append(new_hit.r)

        self._step_cnt += 1
        self._total_step_cnt += 1
        #print(self._step_cnt)

        if (self._total_step_cnt > 2000) & (self._total_step_cnt < 2010): 
            print("I will now save the files ")
            np.savetxt('g_pids.csv', self.record_pid, delimiter=',')
            np.savetxt('g_z.csv', self.record_z, delimiter=',')
            np.savetxt('g_r.csv', self.record_r, delimiter=',')
           # pass 

        if (self._total_step_cnt ==2011) & (self.done_visual == False) : 
            print(self.done_visual, self._total_step_cnt)
            self.my_visualise()
            self.done_visual =True 
            print("it shouldnt happen again")
            #x = 2
       
        if self.num_track_hits > 6:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = [new_hit.z, new_hit.r]

        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=self._point,
                       env_info={
                           'task': self._task,
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
        return f'Point: {self._point}, Goal: {self._goal}'

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
            print("now calling wrap")
            wrap(self.event)
        
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
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']
