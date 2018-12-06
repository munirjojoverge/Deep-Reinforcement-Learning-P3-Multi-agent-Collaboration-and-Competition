#######################################################################
#          Deep Reinforcement Learning Nano-degree - Udacity
#                  Created/Modified on: December 5, 2018
#                      Author: Munir Jojo-Verge
#######################################################################


from unityagents import UnityEnvironment

import numpy as np
import random
import torch
import numpy as np
import os.path
from collections import deque
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime

from ddpg_agent import Agent, ReplayBuffer
from memory import ReplayMemory

class args:
    seed = 777                      # Random seed
    disable_cuda = False            # Disable CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #game = '/media/munirjojo-verge/My Passport/01 - Deep Learning Nanodegree/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux_NoVis/Reacher.x86_64' # Udacity environment        
    game = '/media/munirjojo-verge/My Passport/01 - Deep Learning Nanodegree/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64' # Udacity environment        
    T_max = int(1e3)                # Number of training steps
    max_num_episodes = int(1000)     # Max number of episodes      
    
    hidden_1_size = 96              # Network hidden layer 1 size
    hidden_2_size = 96              # Network hidden layer 2 size
    hidden_3_size = 96              # Network hidden layer 3 size
    
    noise_std = 0.05               # Initial standard deviation of noise added to weights
    
    memory_capacity = int(1e6)      # Experience replay memory capacity
    batch_size = 256*1             # Batch size: Number of memories will be sampled to learn 
    learning_starts_ratio = 1/50   # Number of steps before starting training = memory capacity * this ratio
    learning_frequency = 2    # Steps before we sample from the replay Memory again    
    
    priority_exponent = 0.5         # Prioritised experience replay exponent (originally denoted α)
    priority_weight = 0.4           # Initial prioritised experience replay importance sampling weight
    
    discount = 0.99                 # Discount factor
    
    target_update = int(30)          # Number of steps after which to update target network (Soft update for Actor & Critic)
    tau = 1e-3                      # Soft Update interpolation parameter
    
    reward_clip = 1                 # Reward clipping (0 to disable)
    
    lr_Actor = 1e-3                 # Learning rate - Actor
    lr_Critic = 1e-3                # Learning rate - Critic
    adam_eps = 1e-08                # Adam epsilon (Used for both Networks)
    weight_decay = 0                # Critic Optimizer Weight Decay


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def ddpg(agent, memory, StopWhenSolved = False):
    if os.path.isfile('weights/actor_final.pth'):
        agent.actor_local.load_state_dict(torch.load('weights/actor_final.pth'))
        agent.critic_local.load_state_dict(torch.load('weights/critic_final.pth'))

    
    ######################
    last100_best_scores_deque = deque(maxlen=100)
    scores_global = []
    average_global = []
    min_global = []
    max_global = []
    time_taken = 0
    Start_time = time.time()
    #######################
        
    for i_episode in range(1, agent.args.max_num_episodes):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        
        timestep = 0
        dones = np.zeros(num_agents) 
        while timestep <= agent.args.T_max:                    # while any agents has NOT reached the ball (done)
            timestep += 1 
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished    
                        
            agent.memorize(states, actions, rewards, next_states, dones, memory)
            
            agent.learn(memory,timestep)
                        
            states = next_states
            scores += rewards            
                    
        episode_avg_score = np.mean(scores)                
        scores_global.append(episode_avg_score)
                
        min_global.append(np.min(scores))  
        max_global.append(np.max(scores)) 
        
        last100_best_scores_deque.append(max_global[len(max_global)-1])        
        last100_best_scores_average = np.mean(last100_best_scores_deque)
        
        
        print('\rEpisode {} \tlast 100 avg: {:.2f} \tavg score: {:.2f} '.format(i_episode, last100_best_scores_average, episode_avg_score), end="")
        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'weights/checkpoint_actor_eps' + str(i_episode) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/checkpoint_critic_eps'+ str(i_episode) + '.pth')
            print('\rEpisode {} \tlast 100 avg: {:.2f}'.format(i_episode, last100_best_scores_average)) 
        
        if  (StopWhenSolved and last100_best_scores_average >= 0.5):            
            End_time = time.time()
            time_taken = (End_time - Start_time)/60
            print('\nSolved in {:d} episodes!\tlast100_best_scores_average: {:.2f}, time taken(min): {}'.
                  format(i_episode, last100_best_scores_average, (End_time - Start_time)/60))
            torch.save(agent.actor_local.state_dict(), 'weights/actor_final.pth')
            torch.save(agent.critic_local.state_dict(), 'weights/critic_final.pth')            
            break
     
    return scores_global, average_global, max_global, min_global, time_taken



if __name__=='__main__':

    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    else:
        args.device = torch.device('cpu')

    env = UnityEnvironment(file_name = args.game)
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Construct the Agent
    agent = Agent(args,state_size=state_size, action_size=action_size)

    # Replay memory: We create the Replay Mem outside the Agent so we can share 1 single Mem with all 20 agents. 
    # I will try the simple replay buffer and a more sophisticated priority replay memory.
    #memory = ReplayMemory(args)
    memory = ReplayBuffer(action_size, args.memory_capacity, args.batch_size, args.seed)

    scores_global, average_global, max_global, min_global, time_taken = ddpg(agent, memory, args.max_num_episodes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_global)+1), scores_global)
    plt.plot(np.arange(1, len(average_global)+1), average_global)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(['Episode Avg', 'Last100 Avg'], loc='lower right')
    plt.show()

    fig2 = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(np.arange(1, len(scores_global)+1), scores_global)
    plt.plot(np.arange(1, len(average_global)+1), average_global)
    plt.plot(np.arange(1, len(max_global)+1), max_global)
    plt.plot(np.arange(1, len(min_global)+1), min_global)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(['Episode Avg','Last100 Avg', 'Max', 'Min'], loc='lower right')
    plt.show()

    env.close()

