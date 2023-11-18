import gym
import numpy as np
import time

env = gym.make('FrozenLake-v1')
print(env.observation_space.n) #get number of states
print(env.action_space.n) #get number of actions
env.reset() #reset env to default state
action = env.action_space.sample()  #get a random action
env.render() #render GUI for environment

STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES,ACTIONS))

EPISODES = 1500 #how many times to explore the environment from the beginning
MAX_STEPS = 100 #max num of steps allowed for each episode
LEARNING_RATE = 0.81 #learning rate
GAMMA = 0.96
RENDER = False

epsilon = 0.9 #start with a 90% chance of picking a random action

rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()
        #code to pick action
        if np.random.uniform(0, 1) < epsilon: #we will check if a randomly selected value is less than epsilon
            action = env.action_space.sample() #take random action
        else:
            action = np.argmax(Q[state, :]) #use Q table to pick the best action based on current values   
        next_state, reward, done, info, blah = env.step(action) #take action, notice it returns info 

        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state,:]) - Q[state,action])

        state = next_state
        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break #reached goal
        
print(Q)