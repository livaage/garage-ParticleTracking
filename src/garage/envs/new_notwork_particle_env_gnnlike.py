"""Simple 2D environment containing a point and a goal location."""
import math

import akro
import numpy as np
import circle_fit as cf
import pandas as pd 
from garage import Environment, EnvSpec, EnvStep, StepType
import random 
from gym.spaces import Box
#from visualise_track import visualise 
#from animate_particle import wrap 
#from new_animate_particle import visualise 
from dowel import logger, tabular, CsvOutput
import csv 
import os 

def dip_angle(dr, dz): 
    if dz == 0: 
        dz = 0.01
    dip =  np.tan(dr/dz)

    if math.isnan(dip): 
        print(dr, dz)
    #print(dip)
    return dip

def azimuthal_angle(dx, dy): 
#print(dx, dy)
#x = np.tan(dy, dx)
    if dx ==0: 
        dx = 0.01
    angle = np.arctan2(dy/dx)
    return angle


def estimate_momentum(data): 
    xc,yc,r,_ = cf.least_squares_circle((data))
    #print(pt)
    pt = r*0.01*0.3*3.8  

    return pt 

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi: 
         dphi -= 2*np.pi
    if dphi < -np.pi: 
        dphi += 2*np.pi
    return dphi


def calc_eta(r, z):
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))


#r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
#z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
#pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)

#i = np.where(pids.values.flatten()==-17737)

#my_r = r.values[i]
#my_z = z.values[i]
done_ani = False 

if os.path.exists("garage_outputs.csv"):
  os.remove("garage_outputs.csv")


f = open("garage_outputs.csv", "w")
writer = csv.writer(f)
writer.writerow(["filenumber", "particle_id", "mc_z", "mc_r", "pred_z", "pred_r", "action_z", "action_r"])

#logger.log("testing testing")
event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')


class ParticleEnvGnnLike(Environment):
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
        self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
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
        self._observation_space = akro.Box(low=np.array([-266, 0, -100, -10]), high=np.array([266, 26, 100, 10]), dtype=np.float64)
        self._action_space = akro.Box(low=np.array([-150, -4]),
                                      high=np.array([200,20]),
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
            self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event'+str(self.file_counter)+'.h5')
            print("jumping file")
        self.event = self.event[self.event['sim_pt'] > 2]
        #self.event = self.event[self.event['layer_id']]
        self.event = self.event.loc[self.event.groupby(['particle_id', 'layer_id']).sim_dxy_sig.idxmin().values]
        #subset by the number of hits 
        nh = self.event.groupby('particle_id').agg('count').iloc[:,0]
        # only pick the pids that has a certain number of hits 
        self.event = self.event[self.event['particle_id'].isin(np.array(nh[nh > 7].index))]
        
        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]
        #print(random_particle_id)
        self.original_pid = random_particle_id
        #logger.log('File number:' + str(self.file_counter))
        #logger.log("hello")

        # This relies on an ordered df!  
        start_hit = self.particle.iloc[0,:]
        
        next_hit = self.particle.iloc[1,:]
        self._point = next_hit[['z', 'r']].values 
        self.hit_buffer = [] 
        self.dr_buffer = []
        self.dz_buffer = []
        self.hit_buffer.append([next_hit.x, next_hit.y])

        self.num_track_hits = 0 
        dist = np.linalg.norm(start_hit[['z', 'r']].values - next_hit[['z', 'r']].values)        
        #print(self._point, dist)
        self.state = start_hit.squeeze(axis=0) 
        dist = start_hit[['z', 'r']] - next_hit[['z', 'r']]
        dz =  next_hit.z - start_hit.z
        dr = next_hit.r - start_hit.r
        ##dx = start_hit.x - next_hit.x
        #dy = start_hit.y - next_hit.y
        dphi = calc_dphi(start_hit.sim_phi, next_hit.sim_phi)
        deta = calc_eta(start_hit.r, start_hit.z) - calc_eta(next_hit.r, next_hit.z)


        # self.record_z.append(start_hit.z)
        # self.record_r.append(start_hit.r)
        # self.record_z.append(next_hit.z)
        # self.record_r.append(next_hit.r)
      

        #self.record_file.append(next_hit.r)
        #self.record_pid.append([self.original_pid, self.original_pid])
        # self.record_pid.append(self.original_pid)
        # self.record_pid.append(self.original_pid)
        # self.record_filenumber.append(self.file_counter)
        # self.record_filenumber.append(self.file_counter)
        # self.record_event_counter.append(self.file_counter)
        # self.record_event_counter.append(self.file_counter)
        # np.savetxt(f, next_hit.z, delimiter=",")
        row = pd.DataFrame({'filenumber': [self.file_counter, self.file_counter], 
        'particle_id': [self.original_pid, self.original_pid], 
        'mc_z': [start_hit.z, next_hit.z], 
        'mc_r' : [start_hit.r, next_hit.r], 
        'pred_z': [start_hit.z, next_hit.z], 
        'pred_r': [start_hit.r, next_hit.r], 
        'action_z': [np.nan, np.nan], 
        'action_r': [np.nan, np.nan] })
        row.to_csv(f, mode='a', header=None, index=None)




        #print(dr, dz, dx, dy)

        dip = dip_angle(dr, dz)
        #phi = azimuthal_angle(dx, dy)

        
        observation = np.concatenate((self._point, [dz], [dr]))
        #print(observation)


        
        self.state = next_hit
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

        #print("i am stepping so new count is ", self.new_count)
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        #a = np.clip(a, self.action_space.low, self.action_space.high)
        #print(action[0], a[0])
        #print(a)

        #self._point = np.clip(self._point + a, -266)
        #                      266)
        predicted_point_z = np.clip(self._point[0] + a[0], -266, 266)
        predicted_point_r = np.clip(self._point[1] + a[1], 0, 27)

        #print(a[0], )
        predicted_point = [predicted_point_z, predicted_point_r]

        #print(predicted_point)
        
        if self._visualize:
            print(self.render('ascii'))

        other_hits = self.event[self.event['hit_id']!=self.state.hit_id]
        # it's a big selsarch, converting to list from pandas save an order of magnitude in time,a also just search a small part of the df 
        zlist = other_hits.z.tolist()
        rlist = other_hits.r.tolist() 

        distances = np.sqrt((zlist-predicted_point[0])**2+(rlist - predicted_point[1])**2) 
        index = np.argmin(distances)
        
        new_hit = other_hits.iloc[index, ] 
        #distance_prev_hit = np.sqrt((self.state.r - new_hit.r)**2 + (self.state.z - new_hit.z)**2)
        #distance_prev_hit = [self.state.z - new_hit.z, self.state.r - new_hit.r]
        #mag_dist_prev_hit = np.sqrt(self.state.z-new_hit.z)**2 + (self.state.r-new_hit.r)**2
        self.previous_state = self.state
        self.state = new_hit 
        #self.log_vals() 

        # this is dangerous - relies on ordered df! 
        next_index = self.num_track_hits + 1 
        if next_index > len(self.original_particle) -1: 
            next_index = len(self.original_particle) - 1
        next_hit = self.original_particle.loc[next_index,: ]
        self.hit_buffer.append([predicted_point_z, predicted_point_r])

        #reward given based on how close this new hit was to the next hit in the df 
        distance = np.sqrt((new_hit.z - next_hit.z)**2 + (new_hit.r - next_hit.r)**2)
       # distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
        #print(distance)
        reward = -distance
        #if (mag_dist_prev_hit < 1): 
        #    reward -=100

        self.num_track_hits += 1 
        #print(self.num_track_hits)

        dr = self.state.r - self.previous_state.r
        dz = self.state.z - self.previous_state.z
        #dx = self.state.x - self.previous_state.x 
        #dy = self.state.y - self.previous_state.y
        #dphi = calc_dphi(self.state.sim_phi, self.previous_state.sim_phi)
        #deta = calc_eta(self.state.r, self.state.z) - calc_eta(self.previous_state.r, self.previous_state.z)

        #self.dr_buffer.append(dr)
        #self.dz_buffer.append(dz)
        #m = np.mean(self.dr_buffer)/np.mean(self.dz_buffer)

        #print(dr, dz, dx, dy)
        
        #dip = dip_angle(dr, dz)
        #phi = azimuthal_angle(dx, dy)
        #p = estimate_momentum(self.hit_buffer)

        # self.record_pid.append(self.original_pid)
        # self.record_z.append(predicted_point_z)
        # self.record_r.append(predicted_point_r)
        # self.record_event_counter.append(self.file_counter)
        # self.record_reward.append(reward)
        # self.record_a0.append(a[0])
        # self.record_a1.append(a[1])
        # self.record_filenumber.append(self.file_counter)

        self._step_cnt += 1
        self._total_step_cnt += 1
        # #print(self._step_cnt)

        # if (self._total_step_cnt ==100000): 
        #     print("I will now save the files !!!!!")
        #     np.savetxt('g_pids.csv', self.record_pid, delimiter=',')
        #     np.savetxt('g_z.csv', self.record_z, delimiter=',')
        #     np.savetxt('g_r.csv', self.record_r, delimiter=',')
        #     np.savetxt('g_filenumber.csv', self.record_event_counter, delimiter=',')
        #     np.savetxt('g_reward.csv', self.record_reward, delimiter=',')
        #     np.savetxt('g_a0.csv', self.record_a0, delimiter=',')
        #     np.savetxt('g_a1.csv', self.record_a1, delimiter=',')
        #     np.savetxt('g_files.csv', self.record_filenumber, delimiter=',')

           # pass 

        #if (self._total_step_cnt ==20001) & (self.done_visual == False) : 
        #    print(self.done_visual, self._total_step_cnt)
            #self.my_visualise()
        #    self.done_visual =True 
        #    print("it shouldnt happen again")
            #x = 2
        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [next_hit.z], 
        'mc_r' : [next_hit.r], 
        'pred_z': [predicted_point_z], 
        'pred_r': [predicted_point_r], 
        'action_z': [a[0]], 
        'action_r': [a[1]] })
        row.to_csv(f, mode='a', header=None, index=None)


        stopping = np.mean(np.abs(np.diff(self.dz_buffer[-4:]))) + np.mean(np.abs(np.diff(self.dr_buffer[-4:])))
        if (self.num_track_hits > 6):
        #if a[2] > 0.5:
            #hit_penalty = np.abs(self.num_track_hits - len(self.original_particle))
            #reward = reward - hit_penalty *100
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = [new_hit.z, new_hit.r]
        #self._point = [predicted_point_z, predicted_point_r]
        #distance_to_prev_hit = new_hit[['r', 'z']] - 
        #[np.mean(self.dz_buffer)]

        observation = np.concatenate((self._point, [dz], [dr]))
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
                           'particle_id': self.original_pid, 
                           'actual_actions_z': action[0],
                           'acutal_actions_r': action[1],
                           'predicted_point_z':predicted_point[0], 
                           'predicted_point_r':predicted_point[1],
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

    def dump_summary(self):
        print("dr:   ", "\n dz:    " ) 


    def log_vals(self): 
        logger.log("hello")
        logger.dump_all()
        logger.add_output(CsvOutput('log_folder/table.csv'))

