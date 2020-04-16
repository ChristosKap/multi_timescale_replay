import numpy as np
from functools import partial
import random
#from matplotlib import pyplot as plt

# UTILS

def get_nb_env_params(env_name):
    if env_name == "SunblazeAdaptedHalfCheetah-v0":
        return 3
    elif env_name == "SunblazeAdaptedAnt-v0":
        return 2
    else:
        return 0

def get_eval_schedule_fn(name):
    if name=='linear_density_half_cheetah':
        return eval_linear_density_half_cheetah
    elif name=='linear_gravity_ant':
        return eval_linear_gravity_ant
    
# Functions for generating parameters schedules for different environments

# HalfCheetah

def fixed_params_half_cheetah(t, num_timesteps, power=0.9, gravity=-9.81, friction=0.8):
    return {'gravity': gravity, 'friction': friction, 'power': power}

def linear_gravity_half_cheetah(t, num_timesteps, first=1, last=1, power=0.9, friction=0.8):
    return {'gravity': first + (t / (num_timesteps-1))*(last - first), 'friction': friction, 'power': power}

def random_gravity_half_cheetah(t, num_timesteps, low=-22, high=-2, power=0.9, friction=0.8):
    return {'gravity': random.uniform(low, high), 'power': power, 'friction': friction}

def fluctuating_gravity_half_cheetah(t, num_timesteps, low=-17, high=-7, cycle_length=1000, power=0.9, friction=0.8):
    return {'gravity': (high - low) / 2 * np.sin(2 * np.pi * t / cycle_length) + (high + low) / 2, 'friction': friction, 'power': power}

# Ant
def linear_gravity_ant(t, num_timesteps, first=1, last=1, power=2.5):
    return {'gravity': first + (t / (num_timesteps-1))*(last - first), 'power': power}

def fluctuating_gravity_ant(t, num_timesteps, low=-17, high=-7, cycle_length=1000, power=2.5):
    return {'gravity': (high - low) / 2 * np.sin(2 * np.pi * t / cycle_length) + (high + low) / 2, 'power': power}

def fixed_params_ant(t, num_timesteps, power=2.5, gravity=-9.81):
    return {'gravity': gravity, 'power': power}

# Functions for generating evaluation schedules

def eval_gravity_range_half_cheetah(gravity_range, power=0.9, friction=0.8):
    eval_params = []
    for gravity in gravity_range:
        eval_params.append({'gravity': gravity, 'power': power, 'friction': friction})
    return eval_params

def eval_gravity_range_ant(gravity_range, power=2.5):
    eval_params = []
    for gravity in gravity_range:
        eval_params.append({'gravity': gravity, 'power': power})
    return eval_params
