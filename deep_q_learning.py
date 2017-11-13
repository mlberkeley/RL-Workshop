"""
Chris Lu
November 5, 2017
A basic implementatin of deep q-leqrning for the MLAB workshop.

"""
#OpenAI gym contains a lot of cool environments for reinforcement learning.
import gym

#Keras is a very simple and convenient neural network library. 
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#Numpy is a useful math library for python and has a lot of functionality for multi-dimensional arrays
import numpy as np

#We want to do e-greedy, so we need to occassionally take random actions. We also want to randomly sample from our replay buffer
import random

#makes the gym environment
env = gym.make('CartPole-v0')
action_space = 2 #Number of actions we can take. In CartPole, it is left or right.
observation_space = 4 #Size of the state input we take in. In CartPole, we get like velocity and position and stuff

#CREATING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(10, input_shape = (observation_space,)))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(action_space))
model.compile(loss = 'mse', optimizer = 'adam')
#DONE CREATING NEURAL NETWORK


#Parameters
num_episodes = 1000 
epsilon = 0.25 #For e-greedy
anneal = 0.0025 #How much to decrease epsilon by each episode
exp_buffer = []
batch = 100 #Number of samples to train neural network with
gamma = 0.99 #Discount factor

for i in range(num_episodes):

	obs = env.reset().reshape((1,observation_space)) #Resets the environment and returns a state.

	done = False #Whether or not the episode is finished.

	rAll = 0 #total reward for the episode

	while not done: #While episode is not finished

		if random.random() < epsilon: #epsilon-greedy
			action = random.randint(0, action_space-1) 
		else:
			q_values = model.predict(obs) #Gets the Q-Values from our network
			action = np.argmax(q_values) #The action corresponds to the largest Q-Value

		obs1, reward, done, _ = env.step(action) #Get the next observation, reward, and whether or not you are done from the next time step

		obs1 = obs1.reshape((1,observation_space)) #Reshaping so dimensions fit.

		rAll += reward #increase cumulative reward

		exp_buffer.append((obs,action,reward,obs1,done)) #Add the transition to the experience buffer

		obs = obs1 #Current state = next state

		if i%10 == 0: #Every 10th episode, we render the environment so we can watch it go.
			env.render()

	if len(exp_buffer) > batch:
		minibatch = random.sample(exp_buffer, batch) #We randomly sample from the experience buffer
		inputs = [] #This list will be the observations that we train the Q-Network with
		q_values = [] #This list will be the corresponding labels/target q-values that go with the inputs.
		for m in minibatch:
			o,a,r,o1,d = m #unpacking the tuple
			inputs.append(o) 
			q_vals = model.predict(o)
			"""
			We generate the q-values here during training since we only took one action in this state, so we only update that q-value. 
			In order to ensure that we don't inadvertently affect the other actions, we only change the value of the action we took.
			"""
			if d:
				q_vals[0][a] = r #If done, there is no next state, so it's just the reward.
			else:
				q_vals[0][a] = r + gamma*np.max(model.predict(o1)) #If not done, we get the discounted future rewturn
			q_values.append(q_vals)

		inputs = np.array(inputs).reshape(batch, observation_space) #Reshaping so dimensions fit
		q_values = np.array(q_values).reshape(batch, action_space) #Reshaping so dimensions fit
		model.fit(inputs, q_values, verbose = False)  #Training the model.
 
	epsilon -= anneal #Annealing exploration
	print rAll #Printing reward for the round

	
