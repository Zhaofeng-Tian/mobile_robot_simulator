import numpy as np
from collections import deque
import random

class ReplayBuffer():
    def __init__(self, max_size, obs_shape, state_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.obs_memory = np.zeros((self.mem_size, *obs_shape),dtype=np.float16)
        self.new_obs_memory = np.zeros((self.mem_size, *obs_shape),dtype=np.float16)
        self.lmap_memory = np.zeros((self.mem_size, *obs_shape),dtype=np.float16)
        self.new_lmap_memory = np.zeros((self.mem_size, *obs_shape),dtype=np.float16)
        self.state_memory = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *state_shape),dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, obs, lmap, state, action, reward, obs_, lmap_,state_, done):
        index = self.mem_cntr % self.mem_size
        self.obs_memory[index] = obs
        self.lmap_memory[index] = lmap
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        
        self.new_obs_memory[index] = obs_
        self.new_lmap_memory[index] = lmap_
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        obss = self.obs_memory[batch]
        lmap = self.lmap_memory[batch]
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        obss_ = self.new_obs_memory[batch]
        lmap_ = self.new_lmap_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return obss, lmap, states, actions, rewards, obss_, lmap_ , states_, dones


class QueReplayBuffer():
    def __init__(self, max_size, obs_shape, state_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.obs_memory = deque(maxlen=self.mem_size)
        self.new_obs_memory = deque(maxlen=self.mem_size)
        self.lmap_memory = deque(maxlen=self.mem_size)
        self.state_memory = deque(maxlen=self.mem_size)
        self.new_state_memory = deque(maxlen=self.mem_size)
        self.action_memory = deque(maxlen=self.mem_size)
        self.reward_memory = deque(maxlen=self.mem_size)
        self.terminal_memory = deque(maxlen=self.mem_size)
        self.new_lmap_memory = deque(maxlen=self.mem_size)




        # self.obs_memory = np.zeros((self.mem_size, *obs_shape))
        # self.new_obs_memory = np.zeros((self.mem_size, *obs_shape))
        # self.lmap_memory = np.zeros((self.mem_size, *obs_shape))
        # self.new_lmap_memory = np.zeros((self.mem_size, *obs_shape))
        # self.state_memory = np.zeros((self.mem_size, state_shape))
        # self.new_state_memory = np.zeros((self.mem_size, state_shape))
        # self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, obs, lmap, state, action, reward, obs_, lmap_,state_, done):

        self.obs_memory.append(obs)
        self.lmap_memory.append(lmap)
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
        self.new_obs_memory.append(obs_)
        self.new_lmap_memory.append(lmap_)
        self.new_state_memory.append(state_)
        self.terminal_memory.append(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        obss = np.array(random.sample(self.obs_memory, batch_size))
        lmap = np.array(random.sample(self.lmap_memory,batch_size))
        states = np.array(random.sample(self.state_memory, batch_size))
        actions = np.array(random.sample(self.action_memory, batch_size))
        rewards = np.array(random.sample(self.reward_memory, batch_size))
        obss_ = np.array(random.sample(self.new_obs_memory, batch_size))
        lmap_ = np.array(random.sample(self.new_lmap_memory, batch_size))
        states_ = np.array(random.sample(self.new_state_memory, batch_size))
        dones = np.array(random.sample(self.terminal_memory, batch_size))

        return obss, lmap, states, actions, rewards, obss_, lmap_ , states_, dones
