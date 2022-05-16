#!/usr/bin/env python3
"""This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import tensorflow as tf
import numpy as np

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import AddOrnsteinUhlenbeckNoise, EpsilonGreedyPolicy, AddGaussianNoise
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.algos import DDPG, SAC
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.trainer import TFTrainer
from garage.envs import PointEnv 
#from garage.envs import ParticleEnv, ParticleEnvPrev, ParticleEnvPrevManyFiles, ParticleEnvKalman
from garage.envs import ParticleEnvKalman, ParticleEnvSimple, ParticleEnvGnnLike, OneParticleEnv, ParticlePointEnv
from garage.sampler import LocalSampler, DefaultWorker

@wrap_experiment(archive_launch_repo=False)
def ddpg_partcile(ctxt=None, seed=1):
    """Train DDPG with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = ParticlePointEnv() 

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        exploration_policy = AddOrnsteinUhlenbeckNoise(env.spec,
                                                       policy,
                                                       sigma=np.array([2, 2]))

        #exploration_policy = EpsilonGreedyPolicy(
        #        env_spec=env.spec,
        #        policy=policy,
        #        total_timesteps=10000,
        #        max_epsilon=1.0,
        #        min_epsilon=0.02,
        #        decay_ratio=0.1)


        #exploration_policy = AddGaussianNoise(env.spec,
        #                                  policy,
        #                                  total_timesteps=1000,
         #                                 max_sigma=1,
         #                                 min_sigma=1)



        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sampler = LocalSampler(agents=exploration_policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               #worker_class=DefaultWorker, 
                               worker_class = FragmentWorker
                               #n_workers=1
                               )

        ddpg = SAC(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    sampler=sampler,
                    steps_per_epoch=10,
                    target_update_tau=1e-2,
                    n_train_steps=10,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_policy=exploration_policy,
                    policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                    qf_optimizer=tf.compat.v1.train.AdamOptimizer)

        trainer.setup(algo=ddpg, env=env)

        trainer.train(n_epochs=10, batch_size=100)


ddpg_partcile(seed=1)
