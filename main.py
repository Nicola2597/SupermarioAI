import tensorflow as tf
import time
import gym
import gym_super_mario_bros
# Import the Joypad wrapper
import collections
from nes_py.wrappers import JoypadSpace
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Input
from keras.optimizers import Adam
from tensorflow import *
from tensorflow.python.keras import layers
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import numpy as np

tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#simplify the game controls,remember there are 8 possible actions and create 7 arrays [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,...

class DQNAgent:
  def __init__(self):
       # self.state_size = state_size
        #self.action_size = action_size
        self.hot_encode_azioni = np.array(np.identity(env.action_space.n, dtype=int).tolist())
        self.learning_rate = 0.001
        self.memory = collections.deque(maxlen=2000)
        self.state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
        self.action_size = env.action_space.n  # 8 possible actions
        #self.learning_rate = 0.00025  # for Adam in case we need it
        self.stack_size=4
        ### TRAINING HYPERPARAMETERS
        self.total_episodes = 50  # Total episodes for training
        self.max_steps = 5000  # Max possible steps in an episode otherwise if it gets stuck in the same place forever
        self.batch_size = 64  # Batch size to extract from the memory and to use to avoid learning correlations that are not true but only due to the proximity of the states

        # Exploration parameters for epsilon greedy strategy
        self.explore_start = 1.0  # exploration probability at start
        self.explore_stop = 0.01  # minimum exploration probability
        self.decay_rate = 0.00001  # exponential decay rate for exploration prob

        # Q learning hyperparameters
        self.gamma = 0.9  # Discounting rate

        ### MEMORY HYPERPARAMETERS
        self.pretrain_length = self.batch_size  # Number of experiences stored in the Memory when initialized for the first time
        self.memory_size = 1000000
        self.state_size=4
        # Initialize deque with zero-images one array for each image
        self.stacked_frames = collections.deque([np.zeros((84,84), dtype=int) for i in range(self.stack_size)], maxlen=4)
        #self.optimizer=optimizer
        #q network and target_q_network
        self.q_network = self.model()
        self.target_q_network = self.model()
        #copy q_network parameters into target_q_network parameters
       # self.align_models()
  def align_models(self):
      self.target_q_network.set_weights(self.q_network.get_weights())
  def preprocessing(self,frame):
     self.gray = rgb2gray(frame)

     self.normale_frame = self.gray / 255

     #in order to use the AlexNet convolution we need to reshape the image in 227,227,1,initial image size is 240,256,3
     self.preprocessed_frame = transform.resize(self.normale_frame, [84, 84])

     return self.preprocessed_frame

  def stack_frames(self, stato, is_new_episode):
    # Preprocess frame
    frame = self.preprocessing(stato)
    #if i am not dead then take another 4 of the frame from the state
    if is_new_episode:
        # Clear our stacked_frames
        self.stacked_frames = collections.deque([np.zeros((84,84), dtype=int) for i in range(self.stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        self.stacked_frames.append(frame)
        self.stacked_frames.append(frame)
        self.stacked_frames.append(frame)
        self.stacked_frames.append(frame)

        # Stack the frames
        #example in https://www.geeksforgeeks.org/numpy-stack-in-python/#:~:text=stack()%20is%20used%20for,the%20same%20shape%20and%20dimensions.
        stacked_state = np.stack(self.stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        self.stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(self.stacked_frames, axis=2)

    return stacked_state, self.stacked_frames


   #define a model with AlexNet style for our Q learning agent mind that in future once it is well trained we should copy those parameters into the target Q learning
  def model(self):

      model1 = tf.keras.models.Sequential([
          tf.keras.layers.Conv2D(filters=96, kernel_size=5, strides=4, padding="valid",
                                 activation='relu', input_shape=[84, 84, 4]),
          tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="valid"),
          tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=1, padding='same',
                                 activation='relu'),
          tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
          tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                 activation='relu'),
          tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same',
                                 activation='relu'),
          tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same',
                                 activation='relu'),
          tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(7, activation='softmax')
      ])
      model1.compile(loss = 'mse', optimizer=Adam(lr=self.learning_rate), metrics=["accuracy"])
      return model1



  def act(self,decay_step, q):

        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate * decay_step)
        #exploration
        #print(q.shape)
        if (explore_probability > exp_exp_tradeoff):

            choice=np.random.randint(0,6)



        #exploitation
        else:
            #we need to add a dimension because the neural network will igore the first one and so taking only the 84,84,4 and not just 84,4
            q=tf.convert_to_tensor(np.expand_dims(np.asarray(q).astype(np.float64), axis=0))

            choice1 =np.array(self.q_network.predict(q))


            choice1=np.argmax(choice1,axis=1)
            choice=choice1[0]




        return choice

  def remember(self, state, action, reward, next_state, done):
     self.memory.append((state, action,
                         reward, next_state, done))
  def training(self):
     minibatch=random.sample(self.memory,self.batch_size)
     for state, action, reward, next_state, done in minibatch:
      #check if we have only one step then reward is maximum already
        state=tf.expand_dims(tf.convert_to_tensor(state), 0)
        next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        target = self.q_network.predict(state)

        if done:
          target[0][action]= reward

        else :

          #the target_q_network is only used to estimate the rewards in the next state

          target[0][action]= reward + self.gamma*(np.amax(self.target_q_network.predict(next_state)))



        #use the q parameters reward and action to train the q_network that then would use later to valuate again the target
        self.q_network.fit(state, target, epochs=1, verbose=0)



#TRAINING
#
agent=DQNAgent()
decay_step = 0
print(agent.q_network.summary())

for episode in range(0,agent.total_episodes):
 step=0

 total_reward=0
 state=env.reset()

 #current episode
 state,agent.stacked_frames=agent.stack_frames(state,True)


 while step<agent.max_steps:
    decay_step+=1
    step+=1
    env.render()
    action = agent.act(decay_step,state)


    next_state,reward,done,_= env.step(action)
    reward=reward if step<agent.max_steps else -15

    next_state,agent.stacked_frames=agent.stack_frames(next_state,False)

    agent.remember(state, action, reward, next_state, done)

    state=next_state
    # Add the reward to total reward
    total_reward+=reward
    if done:
        #next_state = np.zeros((84, 84), dtype=np.int)

        #next_state, agent.stacked_frames = agent.stack_frames(next_state, False)
        print("episode: {}/{}, score: {}"
              .format(episode, agent.total_episodes, total_reward))
        # Set step = max_steps to end the episode
        step = agent.max_steps
        agent.align_models()


        break

 if len(agent.memory) > agent.batch_size:
        agent.training()

 if episode % 5 == 0:
     # Saving the model

     path = r"C:\Users\nicol\OneDrive\Documenti\python book/mario2"
     agent.q_network.save_weights(path)
     print('model saved')
env.close()

#test
# agent=DQNAgent()
# path = r"C:\Users\nicol\OneDrive\Documenti\python book/mario2"
# agent.q_network.load_weights(path)
