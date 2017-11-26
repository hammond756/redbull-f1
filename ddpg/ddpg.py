import numpy as np
import random
import argparse
from keras.models import model_from_json, load_model # Model comes from ActorNetwork
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json
import sys

# moved from play method
from keras import backend as K

# import ReplayBuffer.ReplayBuffer
from collections import deque
# from ActorNetwork import ActorNetwork
import math
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.initializers import RandomNormal
# from CriticNetwork import CriticNetwork
# from OU import OU
import timeit

from pytocl.driver import Driver
from pytocl.car import State, Command

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

OU = OU()       #Ornstein-Uhlenbeck Process

# Constants for ActorNetwork
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

# HIDDEN1_UNITS = 300
# HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim],name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)

        # Steering = Dense(1, activation='tanh', kernel_initializer='he_normal')(h1)
        # Acceleration = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(h1)
        # Brake = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(h1)
        # Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        Steering = Dense(1,activation='tanh',init=RandomNormal(1e-4))(h1)
        Acceleration = Dense(1,activation='sigmoid',init=RandomNormal(1e-4))(h1)
        Brake = Dense(1,activation='sigmoid',init=RandomNormal(1e-4))(h1)

        V = merge([Steering,Acceleration,Brake],mode='concat')
        model = Model(input=S,output=V)
        return model, model.trainable_weights, S


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

np.random.seed(1337)

class MyDriver(Driver):

    total_reward = 0.

    train_indicator = 0

    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    vision = False

    EXPLORE = 100000.
    # episode_count = 2000
    # max_steps = 100000
    # reward = 0
    done = False
    step = 0
    epsilon = 1
    # indicator = 0

    actor = None
    critic = None
    buff = None

    s_t = None
    default_speed = 50

    def __init__(self):


        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        K.set_session(sess)

        self.actor = ActorNetwork(sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRA)
        self.critic = CriticNetwork(sess, self.state_dim, self.action_dim, self.BATCH_SIZE, self.TAU, self.LRC)
        self.buff = ReplayBuffer(self.BUFFER_SIZE)    #Create replay buffer

        # Generate a Torcs environment
        # env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

        #Now load the weight
        print("Now we load the weight")
        try:
            self.actor.model.load_weights("ddpg/actormodel.h5")
            self.critic.model.load_weights("ddpg/criticmodel.h5")
            self.actor.target_model.load_weights("ddpg/actormodel.h5")
            self.critic.target_model.load_weights("ddpg/criticmodel.h5")
            print("Weight load successfully")
        except NameError as e:
            print(e)
            sys.exit()
        except:
            print("Cannot find the weight", sys.exc_info())
            sys.exit()

    def reward_function(self, carstate):
        track = carstate.distances_from_edge
        speed = carstate.speed_x
        progress = speed * np.cos(carstate.angle)
        return progress

    def revert_wheel_v_transformation(self, wheel_v):
        DEGREE_PER_RADIANS = 180 / math.pi
        return tuple(v / DEGREE_PER_RADIANS for v in wheel_v)

    def drive(self, carstate):    #1 means Train, 0 means simply Run

        if not np.any(self.s_t):
            corr_wheel_v = self.revert_wheel_v_transformation(carstate.wheel_velocities)
            self.s_t = np.hstack((np.array(carstate.angle, dtype=np.float32),
                                  np.array(carstate.distances_from_edge, dtype=np.float32)/200.,
                                  np.array(carstate.distance_from_center, dtype=np.float32),
                                  np.array(carstate.speed_x, dtype=np.float32)/self.default_speed,
                                  np.array(carstate.speed_y, dtype=np.float32)/self.default_speed,
                                  np.array(carstate.speed_z, dtype=np.float32)/self.default_speed,
                                  np.array(corr_wheel_v, dtype=np.float32)/100.,
                                  np.array(carstate.rpm, dtype=np.float32)))

        # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))


        loss = 0
        self.epsilon -= 1.0 / self.EXPLORE
        a_t = np.zeros([1,self.action_dim])
        noise_t = np.zeros([1,self.action_dim])

        a_t_original = self.actor.model.predict(self.s_t.reshape(1, self.s_t.shape[0]))

        print('a_t_original', a_t_original)

        noise_t[0][0] = self.train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
        noise_t[0][1] = self.train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
        noise_t[0][2] = self.train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

        #The following code do the stochastic brake
        #if random.random() <= 0.1:
        #    print("********Now we apply the brake***********")
        #    noise_t[0][2] = self.train_indicator * max(self.epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        # ob, r_t, done, info = env.step(a_t[0])
        r_t = self.reward_function(carstate)

        corr_wheel_v = self.revert_wheel_v_transformation(carstate.wheel_velocities)

        s_t1 = np.hstack((np.array(carstate.angle, dtype=np.float32),
                              np.array(carstate.distances_from_edge, dtype=np.float32)/200.,
                              np.array(carstate.distance_from_center, dtype=np.float32),
                              np.array(carstate.speed_x, dtype=np.float32)/self.default_speed,
                              np.array(carstate.speed_y, dtype=np.float32)/self.default_speed,
                              np.array(carstate.speed_z, dtype=np.float32)/self.default_speed,
                              np.array(corr_wheel_v, dtype=np.float32)/100.,
                              np.array(carstate.rpm, dtype=np.float32)))

        self.buff.add(self.s_t, a_t[0], r_t, s_t1, self.done)      #Add replay buffer

        #Do the batch update
        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)])

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.GAMMA*target_q_values[k]

        if (self.train_indicator):
            loss += self.critic.model.train_on_batch([states,actions], y_t)
            a_for_grad = self.actor.model.predict(states)
            grads = self.critic.gradients(states, a_for_grad)
            self.actor.train(states, grads)
            self.actor.target_train()
            self.critic.target_train()

        self.total_reward += r_t
        self.s_t = s_t1

        print("Episode", 1, "Step", self.step, "Action", a_t, "Reward", r_t, "Loss", loss)

        self.step += 1

        command = Command()

        command.brake = a_t[0, 2]
        command.accelerator = abs(a_t[0, 1])
        command.steering = a_t[0, 0]

        return command
