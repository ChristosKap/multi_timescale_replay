import random
import numpy as np
import heapq
from collections import deque

def proportional(nsample, buf_sizes):
    T = np.sum(buf_sizes)
    S = nsample
    sample_sizes = np.zeros(len(buf_sizes), dtype=np.int64)
    for i in range(len(buf_sizes)):
        if S < 1:
            break
        sample_sizes[i] = int(round(S * buf_sizes[i] / T))
        T -= buf_sizes[i]
        S -= sample_sizes[i]
    assert sum(sample_sizes) == nsample, str(sum(sample_sizes))+" and "+str(nsample)
    return sample_sizes

def get_replay_buffer(name):
    if name == 'fifo':
        return ReplayBuffer
    elif name == 'reservoir':
        return ReservoirReplayBuffer
    elif name == 'reservoir_with_fifo':
        return ReservoirWithFIFOReplayBuffer
    elif name == 'multi_timescale':
        return MultiTimescaleReplayBuffer
    
class ReplayBuffer(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).
        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            old_data = None
        else:
            old_data = self._storage[self._next_idx]
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        return old_data # used in MultiTimescale buffer
        
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


# Reservoir Buffer
class ReservoirReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        """
        Implements a reservoir sampling  buffer.
        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows,
            experiences are added with random priorities to a priority queue to ensure that buffer
            represents a uniform random sample over all experiences to date.
        """
        self._storage = []
        self._maxsize = size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)
        priority = random.uniform(0, 1)
        if len(self._storage) < self._maxsize:
            heapq.heappush(self._storage, (priority, data))
        elif priority > self._storage[0][0]:
            heapq.heapreplace(self._storage, (priority, data))

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i][1]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)


# Half-reservoir-half-FIFO buffer
class ReservoirWithFIFOReplayBuffer(ReplayBuffer):
    def __init__(self, size, fifo_frac=0.5):
        self._maxsize = size
        assert (fifo_frac > 0) and (fifo_frac < 1)
        self.fifo_frac = fifo_frac
        self._max_fifo_size = round(fifo_frac * size)
        self._max_reservoir_size = size - self._max_fifo_size
        self.fifo_buffer = ReplayBuffer(self._max_fifo_size)
        self.reservoir_buffer = ReservoirReplayBuffer(self._max_reservoir_size)
        
    def __len__(self):
        return len(self.fifo_buffer.storage) + len(self.reservoir_buffer.storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        
        return self.fifo_buffer.storage.extend(self.reservoir_buffer.storage)

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the both the fifo buffer and the reservoir buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        if random.uniform(0,1) < self.fifo_frac:
            self.fifo_buffer.add(obs_t, action, reward, obs_tp1, done)
        else:
            self.reservoir_buffer.add(obs_t, action, reward, obs_tp1, done)
    
    def _encode_sample(self, fifo_idxes, reservoir_idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        buffers = [self.fifo_buffer, self.reservoir_buffer]
        buffer_idxes = [fifo_idxes, reservoir_idxes]
        for i in range(len(buffers)):
            for j in buffer_idxes[i]:
                if i == 1:
                    data = buffers[i]._storage[j][1]
                else:
                    data = buffers[i]._storage[j]
                
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        fifo_batch_size = round(self.fifo_frac * batch_size)
        reservoir_batch_size = batch_size - fifo_batch_size
        fifo_idxes = [random.randint(0, len(self.fifo_buffer._storage) - 1) for _ in range(fifo_batch_size)]
        reservoir_idxes = [random.randint(0, len(self.reservoir_buffer._storage) - 1) for _ in range(reservoir_batch_size)]
        return self._encode_sample(fifo_idxes, reservoir_idxes)

# MTR buffer
class MultiTimescaleReplayBuffer(ReplayBuffer):
    def __init__(self, size, num_buffers, beta=0.5, no_waste=True): 


        print(size, num_buffers)
        self.num_buffers = num_buffers
        self._maxsize_per_buffer = size // num_buffers
        self._maxsize = num_buffers * self._maxsize_per_buffer
        self.beta = beta
        self.no_waste = no_waste
        self.count = 0
        
        if size % num_buffers != 0:
            print("Warning! Size is not divisible by number of buffers. New size is: ", self._maxsize)
        
        self.buffers = []
        for _ in range(num_buffers):
            self.buffers.append(ReplayBuffer(self._maxsize_per_buffer))

        if no_waste:
            self.overflow_buffer = deque(maxlen=self._maxsize)
            
            
    def __len__(self):
        total_length = 0
        for buf in self.buffers:
            total_length += len(buf)
        if self.no_waste:
            total_length += len(self.overflow_buffer)
        return total_length

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        total_storage = []
        for buf in self.buffers:
            total_storage.extend(buf.storage)
        return total_storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        self.count += 1
        data = (obs_t, action, reward, obs_tp1, done)
        popped_data = self.buffers[0].add(*data)
        
        for i in range(1, self.num_buffers):
            #print("buffer ", i)
            #print("popped: ", popped_data)
            if popped_data == None:
                break
            if random.uniform(0, 1) < self.beta:
                popped_data = self.buffers[i].add(*popped_data)
            elif self.no_waste:
                self.overflow_buffer.appendleft(popped_data)
                break
            else:
                break
        if self.no_waste and (self.count > self._maxsize) and (len(self.overflow_buffer) != 0):
            self.overflow_buffer.pop()
            
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        #storage = self.storage
        if self.no_waste:
            assert len(idxes) == (self.num_buffers + 1)
        else:
            assert len(idxes) == self.num_buffers
        for buf_idx in range(len(idxes)):
            for i in idxes[buf_idx]:
                #print(i)
                if buf_idx == 0 and self.no_waste:
                    data = self.overflow_buffer[i]
                else:
                    data = self.buffers[buf_idx - 1].storage[i]
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
            
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        all_idxes = []
        buf_lengths = [len(buf) for buf in self.buffers]
        if self.no_waste:
            buf_lengths.insert(0,len(self.overflow_buffer))

        buffer_batch_sizes = proportional(batch_size, buf_lengths)
        #print(buffer_batch_sizes)
        for i in range(len(buf_lengths)):
            idxes = [random.randint(0, buf_lengths[i] - 1) for _ in range(buffer_batch_sizes[i])]
            all_idxes.append(idxes)
        return self._encode_sample(all_idxes)

    def get_buffer_batch_sizes(self, batch_size):
        buf_lengths = [len(buf) for buf in self.buffers]
        if self.no_waste:
            buf_lengths.insert(0,len(self.overflow_buffer))

        return proportional(batch_size, buf_lengths)


if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    buffer_type = 'multi_timescale'
    buffer_args = {'size': 1000000, 'num_buffers': 20, 'beta': 0.85}
    reservoir = get_replay_buffer(buffer_type)(**buffer_args)
    lengths = []
    sim_length = 5000000
    for i in range(sim_length):
        if i % 100000 == 1:
            print(i)
        reservoir.add(i, i, i, i, False)
        lengths.append(len(reservoir))
    sample = reservoir.sample(10000)
    if buffer_type=='multi_timescale':
        print(len(reservoir.overflow_buffer))
    times = []
    sample_times = []
    if buffer_type == 'reservoir':
        for data in reservoir.storage:
            times.append(sim_length-data[1][0])
    else:
        for data in reservoir.storage:
            times.append(sim_length-data[0])

    # Histogram of experience age
    fig=plt.figure()
    plt.hist(times, bins=range(0,sim_length,10000))
    plt.xlabel('Age', size=16)
    plt.ylabel('Number of experiences', size=16)
    plt.xlim([0, sim_length])
    plt.ylim([0, 11000])
    fig.savefig(buffer_type+"_replay_histogram")

    # Histogram of ages of sample
    plt.figure()
    plt.hist(sample[0], bins=100)

    plt.show()
