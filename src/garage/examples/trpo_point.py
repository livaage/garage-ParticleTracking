from garage import wrap_experiment
from garage.envs import PointEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


@wrap_experiment
def trpo_point(ctxt=None, seed=1):
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = normalize(PointEnv())

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(
            agents=policy,
            envs=env,
            max_episode_length=env.spec.max_episode_length,
            is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=4000)


trpo_point()
