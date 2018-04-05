#!/usr/bin/env python

import tensorflow as tf
import numpy as np

from env import COOP, DEFECT
import logger


class agent(object):

    def __init__(self):
        self.log_percentage_of_action = []
        self.log_average_reward = []
        self.rollout_data = []
        self.old_observation = []

    def act(self, observation, old_info):
        pass

    def train_step(self, obs, acts, advantages):
        pass

    def record_episode_info(self, agent_infolist):
        self.rollout_data.append(agent_infolist)

    def reset_episode_info(self):
        self.rollout_data = []

    def prepare_data(self, rollout_data):
        obs, returns, acts = [], [], []
        self.episodic_returns = []

        for i_episode in rollout_data:
            reward_list = [i_data[1] for i_data in i_episode]
            returns.append(np.array(self.calculate_returns(reward_list)))
            obs.append(np.array([i_data[0] for i_data in i_episode]))
            acts.append(np.array([i_data[3] for i_data in i_episode]))

            self.episodic_returns.append(returns[-1][0])

        return np.concatenate(obs), np.concatenate(acts), \
            np.concatenate(returns)

    def calculate_returns(self, reward_list):
        returns = [reward_list[-1]]
        for i_step in range(1, len(reward_list)):
            returns.append(reward_list[-i_step - 1] + returns[-1])
        return returns[::-1]

    def episode_done(self):
        self.old_observation = None

    def save_npy(self, task, name):
        np.save('./data/' + str(task) + '-' + name + '.npy',
                {'action': self.log_percentage_of_action,
                 'reward': self.log_average_reward})


class selfish_agent(agent):

    def __init__(self, args, session, name_scope):
        agent.__init__(self)

        # initialization
        self.session = session
        self._name_scope = name_scope

        with tf.variable_scope(self._name_scope):
            # build the graph
            self._input = tf.placeholder(tf.float32,
                                         shape=[None, args.input_size])

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self._input, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )
            hidden2 = tf.contrib.layers.fully_connected(
                inputs=hidden1, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden2, num_outputs=args.num_actions,
                activation_fn=None
            )

            # op to sample an action
            self._sample = tf.reshape(tf.multinomial(logits, 1), [])

            # get log probabilities
            log_prob = tf.log(tf.nn.softmax(logits))
            self.log_prob = log_prob

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

    def act(self, observation, old_info):
        # get one action by sampling
        # import pdb; pdb.set_trace()
        act = self.session.run(
            self._sample, feed_dict={self._input: [observation]}
        )  # 0 for coop, 1 for defect
        return act

    def train_step(self):
        obs, acts, advantages = self.prepare_data(self.rollout_data)
        batch_feed = {
            self._input: obs,
            self._acts: acts,
            self._advantages: advantages
        }
        self.session.run(self._train, feed_dict=batch_feed)
        self.log_average_reward.append(np.mean(self.episodic_returns))
        self.log_percentage_of_action.append(np.sum(acts) / np.float(acts.size))

        logger.info(
            'Agent {} have reward: {}'.format(
                self._name_scope, self.log_average_reward[-1]
            )
        )
        '''
        logger.info(
            'Percentage of DEFECT: {}'.format(self.log_percentage_of_action[-1])
        )
        '''

        self.reset_episode_info()


class naive_agent(agent):
    '''
        @brief: always play nice
    '''

    def __init__(self, args, session, name_scope):

        # initialization
        agent.__init__(self)
        self.session = session
        self._name_scope = name_scope

    def act(self, observation, old_info):
        return COOP  # always coop

    def train_step(self):
        obs, acts, advantages = self.prepare_data(self.rollout_data)
        self.log_average_reward.append(np.mean(self.episodic_returns))
        self.log_percentage_of_action.append(np.sum(acts) / np.float(acts.size))

        logger.info(
            'Agent {} have reward: {}'.format(
                self._name_scope, self.log_average_reward[-1]
            )
        )
        '''
        logger.info(
            'Percentage of DEFECT: {}'.format(self.log_percentage_of_action[-1])
        )
        '''
        self.reset_episode_info()


class punishment_agent(agent):
    '''
        @brief: always play nice
    '''

    def __init__(self, args, session, name_scope):

        # initialization
        agent.__init__(self)
        self.session = session
        self._name_scope = name_scope

    def act(self, observation, old_info):
        oppnent_strategy = len(observation) / 2

        if np.any(observation[oppnent_strategy:] == 1):
            return DEFECT  # no-coop if black history found
        else:
            return COOP

    def train_step(self):
        obs, acts, advantages = self.prepare_data(self.rollout_data)
        self.log_average_reward.append(np.mean(self.episodic_returns))
        self.log_percentage_of_action.append(np.sum(acts) / np.float(acts.size))

        logger.info(
            'Agent {} have reward: {}'.format(
                self._name_scope, self.log_average_reward[-1]
            )
        )
        '''
        logger.info(
            'Percentage of DEFECT: {}'.format(self.log_percentage_of_action[-1])
        )
        '''

        self.reset_episode_info()


class adaptive_agent(agent):

    def __init__(self, args, session, name_scope):
        agent.__init__(self)

        # initialization
        self.session = session
        self._name_scope = name_scope
        self.output_size = args.num_actions
        self.unexpected = 0.0

        with tf.variable_scope(self._name_scope + 'selfish'):
            # build the graph
            self.selfish_input = tf.placeholder(
                tf.float32, shape=[None, args.input_size]
            )

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.selfish_input, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )
            hidden2 = tf.contrib.layers.fully_connected(
                inputs=hidden1, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden2, num_outputs=args.num_actions,
                activation_fn=None
            )

            # op to sample an action
            self.selfish_sample = tf.reshape(tf.multinomial(logits, 1), [])

            # get log probabilities
            log_prob = tf.log(tf.nn.softmax(logits))

            # training part of graph
            self.selfish_acts = tf.placeholder(tf.int32)
            self.selfish_advantages = tf.placeholder(tf.float32)

            # get log probs of actions from episode
            indices = tf.range(0, tf.shape(log_prob)[0]) * \
                tf.shape(log_prob)[1] + self.selfish_acts
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

            # surrogate loss
            loss = -tf.reduce_sum(act_prob * self.selfish_advantages)

            # update
            optimizer = tf.train.RMSPropOptimizer(args.learning_rate)
            self.selfish_train = optimizer.minimize(loss)

        with tf.variable_scope(self._name_scope):
            # build the graph
            self._input = tf.placeholder(tf.float32,
                                         shape=[None, args.input_size])

            hidden1 = tf.contrib.layers.fully_connected(
                inputs=self._input, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )
            hidden2 = tf.contrib.layers.fully_connected(
                inputs=hidden1, num_outputs=args.hidden_size,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_normal_initializer()
            )

            logits = tf.contrib.layers.fully_connected(
                inputs=hidden2, num_outputs=args.num_actions,
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

    def act(self, observation, old_info):
        my_action = self.session.run(
            self._sample, feed_dict={self._input: [observation]}
        )
        opponent_previous_ob = old_info[0]
        expected_opponent_act = self.session.run(
            self._sample, feed_dict={self._input: [opponent_previous_ob]}
        )
        # return my_action
        if expected_opponent_act != old_info[1] and (old_info[1] is not None):
            self.unexpected += 1
            # not expected!
            return self.session.run(
                self.selfish_sample,
                feed_dict={self.selfish_input: [observation]}
            )
        else:
            return my_action

    def train_step(self):
        obs, acts, advantages = self.prepare_data(self.rollout_data)
        self.total_episodic_returns = []
        total_advantages = []
        for i_episode in self.rollout_data:
            reward_list = [i_data[5] + i_data[1] for i_data in i_episode]
            total_advantages.append(
                np.array(self.calculate_returns(reward_list))
            )
            self.total_episodic_returns.append(total_advantages[0])
        total_advantages = np.concatenate(total_advantages)

        # import pdb; pdb.set_trace()
        batch_feed = {
            self._input: obs,
            self._acts: acts,
            self._advantages: total_advantages
        }
        self.session.run(self._train, feed_dict=batch_feed)
        batch_feed = {
            self.selfish_input: obs,
            self.selfish_acts: acts,
            self.selfish_advantages: advantages
        }
        self.session.run(self.selfish_train, feed_dict=batch_feed)
        self.log_average_reward.append(np.mean(self.episodic_returns))
        self.log_percentage_of_action.append(
            self.unexpected / np.float(acts.size)
        )

        logger.info(
            'Agent {} have reward: {}'.format(
                self._name_scope, self.log_average_reward[-1]
            )
        )
        logger.info(
            'Percentage of DEFECT: {}'.format(self.log_percentage_of_action[-1])
        )

        logger.info(
            'Total reward: {}'.format(np.mean(self.total_episodic_returns))
        )
        self.unexpected = 0.0
        self.reset_episode_info()
