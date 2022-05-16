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
    data = snapshotter.load('/home/lhv14/garage/src/garage/examples/tf/data/local/experiment/tutorial_vpg_13')
   # data = snapshotter.load('/home/lhv14/garage/src/garage/examples/tf/data/local/experiment/td3_garage_tf_4')
    policy = data['algo'].policy
    env = data['env']
    #env = InferenceParticleEnvGnnLike() 
    #print("THIS IS EVN", env)
    #env = InferenceGnnLike()

    # See what the trained policy can accomplish
    
    pids = np.array([])
    rewards = np.array([]) 
    actions_z = np.array([])
    actions_r = np.array([])
    observations_z = np.array([]) 
    observations_r = np.array([]) 
    actual_actions_z = np.array([])
    actual_actions_r = np.array([])
    predicted_point_z = np.array([])
    predicted_point_r = np.array([])


    for i in range(1000): 
        path = rollout(env, policy, animated=False)
        #print(path['env_infos'])
        #print(path['env_infos'])
        pids = np.append(pids, path['env_infos']['particle_id'].flatten())
        rewards = np.append(rewards, path['rewards'])
        actions_z = np.append(actions_z, path['actions'][:,0])
        #if path['actions'][:,0] > 1: 
        actions_r = np.append(actions_r, path['actions'][:,1]) 
        actual_actions_z = np.append(actual_actions_z, path['env_infos']['actual_actions_z'].flatten())
        actual_actions_r = np.append(actual_actions_r, path['env_infos']['acutal_actions_r'].flatten())
        observations_z = np.append(observations_z, path['observations'][:,0]) 
        observations_r = np.append(observations_r, path['observations'][:,1]) 
       
        predicted_point_z = np.append(predicted_point_z, path['env_infos']['predicted_point_z'].flatten())
        predicted_point_r = np.append(predicted_point_r, path['env_infos']['predicted_point_r'].flatten())

    #print(rewards)
    #np.savetxt('test_rewards.csv', rewards, delimiter=',')
    #np.savetxt("test_actions.csv", actions, delimiter=',')
    #np.savetxt("test_observations.csv", observations, delimiter=',')
    #np.savetxt("test_particle_ids.csv", pids, delimiter=',')
    df = pd.DataFrame({'particle_id':pids.flatten(), 
        'action_z':actions_z, 
        'action_r':actions_r, 
        'actual_action_z': actual_actions_z, 
        'actual_action_r': actual_actions_r,
        'rewards':rewards, 
        'observation_z':observations_z, 
        'observation_r':observations_r, 
        'predicted_point_z':predicted_point_z, 
        'predicted_point_r':predicted_point_r
        })

    print(df) 
    df.to_csv('test_inference_results.csv') 
    print("it is above", np.where(np.abs(actions_z > 1)))
    print("it is above", np.where(np.abs(actions_r > 1)))
    #obtain_evaluation_episodes(policy, env)
