#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the required packages
from gym import wrappers 
from time import time
import time #To calculate execution time
import math
import random  # Used for implementing epsilon policy
import gym  #For using acrobot-v1
import numpy as np 
from collections import deque #For max_memory allocation
from keras.models import Sequential  #For Neural network
from keras.layers import Dense
from keras.optimizers import Adam
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # using plt to plot the total_reward as a function of number of episodes

#These lists are to store episode data, and the accumulated reward dara
episode_data = []  # It will store the episode number
reward_data = []  # It will store the total reward accumulated from random initial state to a terminal state in an episode

#Initialize environment as Acrobat-v1
name_of_env = "Acrobot-v1"

# The discount factor
gamma = 1

# The learning rate of the neural network
learning_rate = 0.001

# The memory to store batches of state-action pairs
size_of_memory = 5000

# The batch size
batch_size = 32

# To implement the epsilon greedy policy
max_exploration = 1.0
min_exploration = 0.02
exploration_decay = 0.995

# Initialising list  to store the total reward history
#reward_history = []

#Defining the number of neurons for each layer in the neural network

no_of_neurons_in_layer1 = 8
no_of_neurons_in_layer2 = 8


# Double Deep Q Learning network architecture defined
class DoubleDeepQLearningNetwork:

    # Architecture of the Deep Neural Network used as the  function approximator
    def __init__(self, observation_space, action_space):
        
        self.exploration_rate = max_exploration
        self.action_space = action_space

        # Memory allocation
        self.memory = deque(maxlen=size_of_memory)
        
        #Defining the first model q

        self.qmodel_a = Sequential()
        self.qmodel_a.add(Dense(no_of_neurons_in_layer1, input_shape=(observation_space,), activation="relu"))
        self.qmodel_a.add(Dense(no_of_neurons_in_layer2, activation="relu"))
        self.qmodel_a.add(Dense(self.action_space, activation="linear"))
        self.qmodel_a.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        
        
        #Defining the target model architecture
        self.qmodel_b = Sequential()

       
        self.qmodel_b.add(Dense(no_of_neurons_in_layer1, input_shape=(observation_space,), activation="relu"))
        self.qmodel_b.add(Dense(no_of_neurons_in_layer2, activation="relu"))
        self.qmodel_b.add(Dense(self.action_space, activation="linear"))
        self.qmodel_b.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        
    #Remember the current state, action, reward, next state and terminal_or_not to do Batch learning of the q model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    #Use the epsilon-greedy policy to decide what action to take on the current state
    def act(self, state):
        #Randomly choose an action
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # Use the learned model to predict the best action by using the argmax(Q(s,a))
        q_values = self.qmodel_a.predict(state)
        return np.argmax(q_values[0])

    #Deep Learning network takes a random sample of batch_size choosen from the memory and learns
    def experience_replay(self):
        
        if len(self.memory) < batch_size:
            return
        # Sample out a random batch of batch_size
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, state_next, terminal in batch:
            #Implementation of core of the algorithm as specified in the pseudocode, updating for Q(s,a) only
            target_values=self.qmodel_a.predict(state)
            if terminal:
                target_values[0][action] = reward
            else:
                qsa=self.qmodel_a.predict(state_next)[0]
                qsb=self.qmodel_b.predict(state_next)[0]
                target_values[0][action]=reward+ gamma*qsb[np.argmax(qsa)]
            # DDQNN learns the calculated Q values for the state
            self.qmodel_a.fit(state, target_values, epochs=1, verbose=0)

        # Exploration Decay
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(min_exploration, self.exploration_rate)


# Simulating Episodes
def acrobot():
    env = gym.make(name_of_env).env  #Initializing the environment as acrobot
    

    observation_space = env.observation_space.shape[0] # No of state values=6
    action_space = env.action_space.n  # No of Actions possible=3
    ddqn = DoubleDeepQLearningNetwork(observation_space, action_space) # Initialising the DDQN
    episode = 0  #Episode no
    total_training_steps = 0  #Total number of training steps

    while episode <= 100:
        episode += 1  
        episode_data.append(episode)  # Storing the episode number to plot
        total_reward = 0  # Variable to store the total accumulated reward of the current episode
        state = env.reset()  # Picks up a random current state from the env

        state = np.reshape(state, [1, observation_space])  # Picks up the state observation space
        step = 0  # stores the count of number of steps in current episode
        while True:
            step += 1
            env.render()  #renders the moving arms on the display to visualize
            action = ddqn.act(state)  # Chooses the best action possible on the current state using the DDQN

            state_next, reward, terminal, info = env.step(action) #takes the action generated above and calculates nextstate,
            #reward,terminal_or_not

            # As per the implementation of Acrobot-v1 in git the reward is -1 if state is not terminal and 0 if terminal

            reward = reward if not terminal else -reward

            # Accumulating the total reward
            total_reward += reward

            state_next = np.reshape(state_next, [1, observation_space])

            ddqn.remember(state, action, reward, state_next, terminal)  # remembering the values in memory
            state = state_next  # updating current state by the next state

            # End of Episode if the terminal state is reached
            if terminal:

                print('The episode is: ' + str(episode) + ",    Exploration is: " + str(
                    ddqn.exploration_rate) + ",     No of steps is: " + str(step) + '      ,Total Reward is: ' + str(total_reward))
                reward_data.append(total_reward)  # storing total_reward accumulated till the end of episode to plot
                reward_history.append(total_reward)  # Storing the total_ accumulated reward in the history

                total_training_steps += step  #Accumulate the no of training steps
                break
            else:

                ddqn.experience_replay()  # DDQN  will continue training if not terminal terminal state



if __name__ == "__main__":
    acrobot()
    axes = plt.gca()
    axes.set_xlim([1, 100])
    plt.plot(episode_data, reward_data)
    plt.xlabel('Episode number')
    plt.ylabel('Reward obtained')
    plt.show()
    print('Maximum best total reward obtained was ' + str(np.max(reward_history)))
    
    
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:




