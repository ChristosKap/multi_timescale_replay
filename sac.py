import sys
import time
import multiprocessing
from collections import deque
import warnings
import os
import cloudpickle
from functools import partial

import numpy as np
import tensorflow as tf
import replay_buffers

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
#from stable_baselines.sac.policies import SACPolicy
from policies import SACPolicy, squashed_gaussian_likelihood
from stable_baselines import logger


def get_vars(scope):
    """
    Alias for get_trainable_vars
    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)

class SAC(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
    :param env_adaptation_fn: (callable) Function of timestep and total timesteps that returns environment parameters. None if not using an adaptive environment.
    :param env_adaptation_interval: (int) how frequently to update environment parameters (in timesteps)
    :param eval_env: (Gym environment) Environment to evaluate agent on.
    :param eval_interval: (int) Evaluate agent every 'eval_interval' episodes.
    :param eval_env_params: (list of dicts) List of environment parameter settings for evaluating the agent.
    :param num_eps_per_eval: (int) run each evaluation parameter setting for 'num_eps_per_eval' episodes.
    :param replay_buffer_type: (string) See options in buffers.py.
    :param replay_buffer_params: (dict) parameters for replay buffer.
    :param irm_replay: (bool) Whether IRM loss is used to encourage invariance across time. Can only be used with MultiTimescale buffer.
    :param irm_pol_coef: (float) Coefficient of IRM loss on policy.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 env_adaptation_fn=None, env_adaptation_interval=1000, eval_env=None,
                 eval_interval=1000, eval_env_params=None, num_eps_per_eval=1,
                 replay_buffer_type='fifo', replay_buffer_params={}, irm_replay=False,
                 irm_pol_coef=1.0):

        super(SAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        self.env_adaptation_fn = env_adaptation_fn
        self.env_adaptation_interval = env_adaptation_interval
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_env_params = eval_env_params
        self.num_eps_per_eval = num_eps_per_eval

        self.replay_buffer_type = replay_buffer_type
        self.replay_buffer_params = replay_buffer_params
        self.irm_replay = irm_replay
        self.irm_pol_coef = irm_pol_coef
     
        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                replay_buffer_fn = replay_buffers.get_replay_buffer(self.replay_buffer_type)
                self.replay_buffer = replay_buffer_fn(**self.replay_buffer_params)

                if self.replay_buffer_type == 'multi_timescale' or self.replay_buffer_type == 'refer_multi_timescale':
                    self.num_mtr_buffers = self.replay_buffer_params['num_buffers']
                else:
                    self.num_mtr_buffers = 1
                    
                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    # Placeholder for splitting observations into environments for IRM loss
                    self.irm_env_split_idxes = tf.placeholder(tf.int64, shape=(self.num_mtr_buffers + 1,), name="irm_env_split_idxes")
                    
                    # IRM dummy variables
                    self.dummy_pol = tf.constant(1.0, name="dummy_pol")
                    
                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                        
                    self.value_target = value_target
                        
                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    ) 
                    
                    # Compute Q-Function loss
                    qf1_unreduced_loss = 0.5 * (q_backup - qf1) ** 2
                    qf2_unreduced_loss = 0.5 * (q_backup - qf2) ** 2
                    qf1_loss = tf.reduce_mean(qf1_unreduced_loss)
                    qf2_loss = tf.reduce_mean(qf2_unreduced_loss)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - tf.squeeze(qf1_pi))
                    
                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss
                    
                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_unreduced_loss = 0.5 * (value_fn - v_backup) ** 2
                    value_loss = tf.reduce_mean(value_unreduced_loss)

                    values_losses = qf1_loss + qf2_loss + value_loss
                            
                    def squared_grad_norm_fn(dummy):
                        return lambda y: tf.reduce_mean(tf.square(tf.gradients(y, dummy)))
                    
                    # IRM policy loss
                    unreduced_policy_losses = self.ent_coef * self.policy_tf.get_irm_logp_pi(dummy=self.dummy_pol) - tf.squeeze(qf1_pi)
                    # Ignore first element in split which corresponds to overflow buffer
                    policy_loss_by_env = tf.split(self.ent_coef * self.policy_tf.get_irm_logp_pi(dummy=self.dummy_pol) - tf.squeeze(qf1_pi), self.irm_env_split_idxes)[1:] 
                    policy_loss_grad_mags_by_env = [squared_grad_norm_fn(self.dummy_pol)(split) for split in policy_loss_by_env]
                    policy_irm_loss = self.irm_pol_coef * tf.reduce_mean(policy_loss_grad_mags_by_env)
                    if self.irm_replay and self.irm_pol_coef is not 0:
                        policy_loss += policy_irm_loss
                                            
                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))
                    
                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")
            
                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]
                    policy_train_ops = [policy_train_op]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies(policy_train_ops):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]
                        if self.irm_replay:
                            if self.irm_pol_coef is not 0:
                                self.infos_names += ['irm_policy_loss']
                                self.step_ops += [policy_irm_loss]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if self.irm_replay:
                        if self.irm_pol_coef is not 0:
                            tf.summary.scalar('irm policy loss', policy_irm_loss)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }
            
        if self.irm_replay:
            irm_splits = self.replay_buffer.get_buffer_batch_sizes(self.batch_size) 
            feed_dict[self.irm_env_split_idxes] = irm_splits
            
        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]
        
        return_values = (policy_loss, qf1_loss, qf2_loss, value_loss, entropy)

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[8:10]
            return_values += (ent_coef_loss, ent_coef)

        if self.irm_replay:
            if 'irm_policy_loss' in self.infos_names:
                return_values += (values[self.infos_names.index('irm_policy_loss') + 3],)
            
        return return_values

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            if self.env_adaptation_fn is not None:
                self.env.set_env_params(self.env_adaptation_fn(0, num_timesteps=total_timesteps))
            obs = self.env.reset()

            if self.eval_env is not None:
                if self.eval_env_params is not None:
                    self.eval_env.set_env_params(self.eval_env_params[0])
                eval_obs = self.eval_env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if (self.env_adaptation_fn is not None) and (step % self.env_adaptation_interval == 0):
                    env_params = self.env_adaptation_fn(step, num_timesteps=total_timesteps)
                    self.env.set_env_params(env_params)
                    print("env_params: ", env_params)
                    
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)
                    
                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                
                if (self.eval_env is not None) and (num_episodes  % self.eval_interval) and done:
                    eval_ep_reward_means = []
                    eval_ep_reward_stds = []
                    eval_mean_ep_lengths = []
                    for eval_num in range(len(self.eval_env_params)):
                        curr_eval_env_params = self.eval_env_params[eval_num]
                        self.eval_env.set_env_params(curr_eval_env_params)
                        eval_ep = 0
                        eval_ep_rewards = []
                        eval_ep_lengths = []
                        while eval_ep < self.num_eps_per_eval:
                            t_eval = 0
                            eval_ep_reward = 0
                            eval_done = False
                            while not eval_done:
                                eval_action,_ = self.predict(eval_obs)
                                eval_obs, eval_r, eval_done, eval_info = self.eval_env.step(eval_action)
                                eval_ep_reward += eval_r
                                t_eval += 1
                            eval_ep_lengths.append(t_eval)
                            eval_ep_rewards.append(eval_ep_reward)
                            eval_ep += 1
                        eval_ep_reward_means.append(np.mean(eval_ep_rewards))
                        eval_ep_reward_stds.append(np.std(eval_ep_rewards))
                        eval_mean_ep_lengths.append(np.mean(eval_ep_lengths))
                        
                    

                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    logger.logkv("replay_buffer_size", len(self.replay_buffer))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    if self.env_adaptation_fn is not None:
                        for (k, v) in env_params.items():
                            logger.logkv("env_params/"+k, v)
                    if self.eval_env is not None:
                        for eval_num in range(len(eval_ep_reward_means)):
                            logger.logkv("eval_"+str(eval_num)+" ep_rewmean", eval_ep_reward_means[eval_num])
                            logger.logkv("eval_"+str(eval_num)+" ep_rewstd", eval_ep_reward_stds[eval_num])
                            logger.logkv("eval_"+str(eval_num)+" ep_lenmean", eval_mean_ep_lengths[eval_num])
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "env_adaptation_fn": self.env_adaptation_fn,
            "env_adaptation_interval": self.env_adaptation_interval,
            "eval_interval": self.eval_interval,
            "eval_env_params": self.eval_env_params,
            "num_eps_per_eval": self.num_eps_per_eval,
            "replay_buffer_type": self.replay_buffer_type,
            "replay_buffer_params": self.replay_buffer_params,
            "irm_replay": self.irm_replay,
            "irm_pol_coef": self.irm_pol_coef,
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save) #, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, eval_env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        #data, params = cls._load_from_file(load_path, custom_objects=custom_objects)
        data, params = cls._load_from_file_cloudpickle(load_path)
        
        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.set_eval_env(eval_env)
        model.setup_model()

        model.load_parameters(params)

        return model

    def set_eval_env(self, eval_env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the evaluation environment.
        :param eval_env: (Gym Environment) The environment for evaluating a policy
        """
        if eval_env is None and self.eval_env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif eval_env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == eval_env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == eval_env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        if self._requires_vec_env:
            assert isinstance(eval_env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not self.policy.recurrent or self.n_envs == eval_env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            #self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(eval_env, VecEnv):
                if eval_env.num_envs == 1:
                    eval_env = _UnvecWrapper(eval_env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            #self.n_envs = 1

        self.eval_env = eval_env

    @staticmethod
    def _load_from_file_cloudpickle(load_path):
        """Legacy code for loading older models stored with cloudpickle
        :param load_path: (str or file-like) where from to load the file
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file_:
                data, params = cloudpickle.load(file_)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params
