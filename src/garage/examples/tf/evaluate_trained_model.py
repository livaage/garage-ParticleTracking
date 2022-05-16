import numpy as np
import pandas as pd


# Load the policy and the env in which it was trained
from garage.experiment import Snapshotter
from garage import rollout, obtain_evaluation_episodes
import tensorflow as tf # optional, only for TensorFlow as we need a tf.Session
#from garage.envs import InferenceParticleEnvGnnLike, InferenceGnnLike  
from garage.envs import ParticleEnvGnnLike
snapshotter = Snapshotter()
with tf.compat.v1.Session(): # optional, only for TensorFlow
    data = snapshotter.load('/home/lhv14/garage/src/garage/examples/tf/data/local/experiment/tutorial_vpg_64', itr='last')
   # data = snapshotter.load('/home/lhv14/garage/src/garage/examples/tf/data/local/experiment/td3_garage_tf_4')
    policy = data['algo'].policy
    env = data['env']
    #env = InferenceParticleEnvGnnLike() 
    #print("THIS IS EVN", env)
    #env = InferenceGnnLike()

    # See what the trained policy can accomplish

    
    keys = ['filenumber', 'particle_id', 'mc_z', 'mc_r', 'pred_z', 'pred_r', 'action_z', 'action_r']
    #res = {key: [] for key in keys} 
    res = dict([(key, []) for key in keys]) 
    for i in range(10): 
      path = rollout(env, policy, animated=False)
      for key in path['env_infos'].keys(): 
        res[key].extend(path['env_infos'][key].flatten())
          #   res[key] = 0
        #pids = np.append(pids, path['env_infos']['particle_id'].flatten())
        #rewards = np.append(rewards, path['rewards'])
        #actions_z = np.append(actions_z, path['actions'][:,0])
        #if path['actions'][:,0] > 1: 
        #actions_r = np.append(actions_r, path['actions'][:,1]) 
        #actual_actions_z = np.append(actual_actions_z, path['env_infos']['actual_actions_z'].flatten())
df = pd.DataFrame(res)
df.to_csv('inference_resuts.csv')
