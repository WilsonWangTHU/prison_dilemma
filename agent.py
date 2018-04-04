#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from env import COOP, DEFECT

class agent(object):
    def act(self, observation):
        pass

    def train_step(self, obs, acts, advantages):
        pass

    def record_episode_info(self, observation):
        pass

    def reset_episode_info(self):
        pass

    def prepare_data(self, rollout_data):
        obs, returns, acts = [], [], []
        self.episodic_returns = []

        for i_episode in rollout_data:
            reward_list = [i_data[1] for i_data in i_episode]
            returns.append(np.array(self.calculate_returns(reward_list)))
            obs.append(np.array([i_data[0] for i_data in i_episode]))
            acts.append(np.array([i_data[3] for i_data in i_episode]))

            self.episodic_returns.append(returns[0])

        return np.concatenate(obs), np.concatenate(acts), \
            np.concatenate(returns)

    def calculate_returns(self, reward_list):
        returns = [reward_list[-1]]
        for i_step in range(1, len(reward_list)):
            returns.append(reward_list[-i_step - 1] + returns[-1])
        return returns[::-1]

class selfish_agent(agent):
    '''
        @brief: the selfish agent
    '''

    def __init__(self, args, session, name_scope):

        # initialization
        self.session = session
        self._name_scope = name_scope
        self.rollout_data = []

        with tf.variable_scope(self._name_scope):
            # build the graph
            self._input = tf.placeholder(tf.float32,
                                         shape=[None, args.input_size])

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self._input, num_outputs=args.hidden_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer()
            )

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden1, num_outputs=args.num_actions,
                activation_fn=None
            )

            # op to sample an action
            self._sample = tf.reshape(tf.multinomial(logits, 1), [])

            # get log probabilities
            log_prob = tf.log(tf.nn.softmax(logits))

            # training part of graph
            self._acts = tf.placeholder(tf.int32)
            self._advantages = tf.placeholder(tf.float32)

            # get log probs of actions from episode
            indices = tf.range(0, tf.shape(log_prob)[0]) * \
                tf.shape(log_prob)[1] + self._acts
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

            # surrogate loss
            loss = -tf.reduce_sum(act_prob * self._advantages)

            # update
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
            self._train = optimizer.minimize(loss)

    def act(self, observation):
        # get one action by sampling
        # print observation
        return self.session.run(
            self._sample, feed_dict={self._input: [observation]}
        )  # 0 for coop, 1 for defect

    def train_step(self):
        obs, acts, advantages = self.prepare_data(self.rollout_data)
        batch_feed = {
            self._input: obs,
            self._acts: acts,
            self._advantages: advantages
        }
        self.session.run(self._train, feed_dict=batch_feed)
        # logging the reward function
        print(np.mean(self.episodic_returns))
        print(acts)

        self.reset_episode_info()

    def record_episode_info(self, agent_infolist):
        self.rollout_data.append(agent_infolist)

    def reset_episode_info(self):
        self.rollout_data = []



class naive_agent(agent):
    '''
        @brief: always play nice
    '''

    def __init__(self, args, session, name_scope):

        # initialization
        self.session = session

    def act(self, observation):
        return COOP  # always coop

    def train_step(self, obs, acts, advantages):
        pass


class punishment_agent(agent):
    '''
        @brief: always play nice
    '''

    def __init__(self, args, session, name_scope):

        # initialization
        self.session = session

    def act(self, observation):
        oppnent_strategy = len(observation) / 2

        # print np.any(observation[oppnent_strategy:] == 1)
        # print observation
        # print '----'
        if np.any(observation[oppnent_strategy:] == 1):
            return DEFECT  # no-coop if black history found
        else:
            return COOP

    def train_step(self, obs, acts, advantages):
        pass
