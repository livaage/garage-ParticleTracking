from garage import wrap_experiment
from garage.envs import TestPointEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import ContinuousMLPPolicy
from garage.trainer import TFTrainer
from garage.tf.algos import TD3
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.replay_buffer import PathBuffer
from garage.envs import ParticleEnvGnnLike
import tensorflow as tf 

@wrap_experiment
def trpo_point(ctxt=None, seed=1):
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = ParticleEnvGnnLike()

        policy = ContinuousMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True)



        qf = ContinuousMLPQFunction(name='ContinuousMLPQFunction',
                                    env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    action_merge_layer=0,
                                    hidden_nonlinearity=tf.nn.relu)

        qf2 = ContinuousMLPQFunction(name='ContinuousMLPQFunction2',
                                     env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     action_merge_layer=0,
                                     hidden_nonlinearity=tf.nn.relu)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        algo = TD3(env_spec=env.spec,
                    policy=policy,
                    sampler=sampler,
                    discount=0.99,
                    qf=qf,
                    qf2=qf2,
                    replay_buffer=replay_buffer,)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000)


trpo_point()
