from gym import ObservationWrapper
from gym.spaces import Box
import gym
import roboschool
import adapted_envs
import numpy as np
import sunblaze_envs

class AdaptiveEnvWrapper(ObservationWrapper):
    """ Adds extra environment variables to observation and enables setting of extra variables """
    def __init__(self, env, num_env_params):
        super(AdaptiveEnvWrapper, self).__init__(env)
        obs_space = env.observation_space
        obs_shape = obs_space.shape
        assert len(obs_shape) == 1
        assert isinstance(obs_space, Box)

        self.num_env_params = num_env_params
        new_obs_low = np.concatenate([np.full((num_env_params,), -np.inf),
                                      obs_space.low])
        new_obs_high = np.concatenate([np.full((num_env_params,), np.inf),
                                       obs_space.high])
        self.observation_space = Box(low=new_obs_low, high=new_obs_high)
        self.env_params = None
        
    def observation(self, observation):
        extra_obs = [v for (k, v) in self.env_params.items()]
        return np.concatenate([extra_obs, observation])

    def set_env_params(self, new_env_params):
        assert len(new_env_params) == self.num_env_params
        self.env_params = new_env_params
        self.env.env.set_env_params(new_env_params)
        # set new vars in env

if __name__=="__main__":
    env = sunblaze_envs.make("SunblazeAdaptedHalfCheetah-v0")
    print(env.observation_space.shape)
    env = AdaptiveEnvWrapper(env, 3)
    print(env)
    print(env.observation_space.shape)
    print(env.observation_space.low)
    print(env.observation_space.high)
