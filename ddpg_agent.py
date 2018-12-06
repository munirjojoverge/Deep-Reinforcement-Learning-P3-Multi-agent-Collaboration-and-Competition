#######################################################################
#          Deep Reinforcement Learning Nano-degree - Udacity
#                  Created/Modified on: November 4, 2018
#                      Author: Munir Jojo-Verge
#######################################################################

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model2 import Actor, Critic
#from model import Actor, Critic

from memory import *

import torch    
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, args, state_size, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            
        """
        self.args = args
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(self.args.seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, self.args.seed, fc1_units=self.args.hidden_1_size, fc2_units=self.args.hidden_2_size, fc3_units=self.args.hidden_3_size).to(device)
        self.actor_target = Actor(state_size, action_size, self.args.seed, fc1_units=self.args.hidden_1_size, fc2_units=self.args.hidden_2_size, fc3_units=self.args.hidden_3_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.args.lr_Actor , eps=self.args.adam_eps)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, self.args.seed, fc1_units=self.args.hidden_1_size, fc2_units=self.args.hidden_2_size, fc3_units=self.args.hidden_3_size).to(device)
        self.critic_target = Critic(state_size, action_size, self.args.seed, fc1_units=self.args.hidden_1_size, fc2_units=self.args.hidden_2_size, fc3_units=self.args.hidden_3_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.args.lr_Critic, eps=self.args.adam_eps, weight_decay=self.args.weight_decay) 

        # Noise process
        self.noise = OUNoise(action_size, self.args.seed, sigma=self.args.noise_std)

        self.learning_starts = int(args.memory_capacity * args.learning_starts_ratio) 
        self.learning_frequency = args.learning_frequency       
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, LEARNING_BATCH_SIZE, seed)
    
    def memorize(self, states, actions, rewards, next_states, dones, memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""        
        # Save experience / reward
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):        
            #memory.add(state, action, reward, done)
            memory.add(state, action, reward, next_state, done)

       
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, memory, timestep):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            
        """
        # Learn, if enough samples are available in memory and after "learning_frequency" steps since we last learnt
        # if memory.num_memories > self.learning_starts and timestep % self.learning_frequency == 0:           
        #   idxs, states, actions, rewards, next_states, dones, _ = memory.sample(self.args.batch_size)
                    
        if len(memory) > self.learning_starts and timestep % self.learning_frequency == 0:           
            states, actions, rewards, next_states, dones = memory.sample()
                        
            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.args.discount * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)        
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()        
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.args.reward_clip)
            self.critic_optimizer.step()
            


            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            

            # memory.update_priorities(idxs, actor_loss.detach())  # Update priorities of sampled transitions
            

            # ----------------------- update target networks ----------------------- #
            # if timestep % self.args.target_update == 0:
            # Every time there is a leartning process happening, let's update
            self.soft_update(self.critic_local, self.critic_target, self.args.tau)
            self.soft_update(self.actor_local, self.actor_target, self.args.tau)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, LEARNING_BATCH_SIZE, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            LEARNING_BATCH_SIZE (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.LEARNING_BATCH_SIZE = LEARNING_BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=int(self.LEARNING_BATCH_SIZE))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)