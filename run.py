import gym
import numpy as np
import roboschool
import sunblaze_envs
import adapted_envs
import env_schedules
from functools import partial
import copy
import os

#from stable_baselines.sac.policies import MlpPolicy
from policies import MlpPolicy
from dummy_vec_env import DummyVecEnv
from sac import SAC
from stable_baselines import logger
from mtr_utils import AdaptiveEnvWrapper

num_modifiers = 3 # 3 for HalfCheetah, 2 for Ant
env = sunblaze_envs.make('SunblazeAdaptedHalfCheetah-v0')
env = AdaptiveEnvWrapper(env, num_modifiers)
print(env)
env = DummyVecEnv([lambda: env])
eval_env = copy.deepcopy(env)
eval_env_params = env_schedules.eval_gravity_range_half_cheetah([-7, -9.5, -12, -14.5, -17])
#eval_env_params = env_schedules.eval_gravity_range_ant([-7, -9.5, -12, -14.5, -17])
eval_interval = 100
num_eps_per_eval = 1
env_adaptation_fn = partial(env_schedules.linear_gravity_half_cheetah, first=-7.0, last=-17.0)
replay_buffer_type = 'multi_timescale'
replay_buffer_params = {'size': 1000000, 'num_buffers': 20, 'beta': 0.85, 'no_waste': True}
total_timesteps = 5000000
train_freq = 1
irm_replay = False
irm_pol_coef = 0.1
log_path = 'mtr_test_no_irm'
log_interval = 10
logger.configure(log_path, format_strs=['stdout', 'csv', 'log'])
lr = 3e-4
learning_starts = 100
ent_coef = 'auto'

model = SAC(MlpPolicy, env,
            verbose=1,
            batch_size=256,
            learning_rate=lr,
            env_adaptation_fn = env_adaptation_fn,
            eval_env=eval_env,
            eval_env_params=eval_env_params,
            eval_interval=eval_interval,
            num_eps_per_eval=num_eps_per_eval,
            replay_buffer_type=replay_buffer_type,
            replay_buffer_params=replay_buffer_params,
            train_freq=train_freq,
            irm_replay=irm_replay,
            irm_pol_coef=irm_pol_coef,
            learning_starts=learning_starts,
            ent_coef=ent_coef,
            policy_kwargs={'layers': [256, 256]})

model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
