#!/usr/bin/env python

import numpy as np
import logger

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


class maze_env(object):
    '''
        1 for yourself, -1 for opponent, 2 for the foot
        0 0 0 0 0
        0 0 0 b 0
        A 0 1 0 B
        0 a 0 0 0
        0 0 0 0 0

        you can either eat your share (+2) or eat the joint prey togather (+1.5)
        But eat at the same time, then it's (+0) for both side
    '''

    def __init__(self, args):
        self.maze_size = 5
        self.history_length = args.history_length
        self.game_length = 2
        # assert self.history_length < self.game_length
        self.agent_move = {
            LEFT: np.array([0, -1]),
            RIGHT: np.array([0, 1]),
            UP: np.array([-1, 0]),
            DOWN: np.array([1, 0])
        }

    def reset(self):
        self.target_pos = np.array([2, 2], dtype=np.int)
        self.target_one_pos = np.array([3, 1], dtype=np.int)
        self.target_two_pos = np.array([1, 3], dtype=np.int)

        self.agent_one_pos = np.array([2, 0], dtype=np.int)
        self.agent_two_pos = np.array([2, 4], dtype=np.int)

        self.current_time = 0
        self.generate_obs()

        player_one_ob = np.array(self.player_one_state + self.player_two_state)
        player_two_ob = np.array(self.player_two_state + self.player_one_state)
        return [[player_one_ob, 0, False],
                [player_two_ob, 0, False]]

    def generate_obs(self):
        # [2, 2], [agent 2], [3, 1]
        self.player_one_state = \
            (self.target_pos - self.agent_one_pos).tolist() + \
            (self.agent_two_pos - self.agent_one_pos).tolist() + \
            (self.target_one_pos - self.agent_one_pos).tolist()

        # use non-global observation
        self.player_two_state = \
            (self.agent_two_pos - self.target_pos).tolist() + \
            (self.agent_two_pos - self.agent_one_pos).tolist() + \
            (self.agent_two_pos - self.target_two_pos).tolist()

    def step(self, player_one_action, player_two_action):
        self.current_time += 1
        # print player_one_action

        # make the move for each agent
        # logger.error('Before')
        # self.render()
        self.agent_one_pos += self.agent_move[player_one_action]
        self.agent_two_pos += -self.agent_move[player_two_action]

        self.agent_one_pos = np.maximum(
            np.minimum(self.agent_one_pos, self.maze_size - 1), 0
        )
        self.agent_two_pos = np.maximum(
            np.minimum(self.agent_two_pos, self.maze_size - 1), 0
        )

        self.generate_obs()
        player_one_ob = np.array(self.player_one_state + self.player_two_state)
        player_two_ob = np.array(self.player_two_state + self.player_one_state)

        if_eat_one = self.if_eat(self.agent_one_pos, self.target_one_pos)
        if_eat_two = self.if_eat(self.agent_two_pos, self.target_two_pos)

        if_eat_both = self.if_eat(self.agent_one_pos, self.target_pos) or \
            self.if_eat(self.agent_two_pos, self.target_pos)

        # generate the reward from reward table
        if if_eat_both:
            reward_one, reward_two = 2, 2
            done = True
        elif if_eat_one and if_eat_two:
            reward_one, reward_two = 0, 0
            done = True
        elif if_eat_one:
            reward_one, reward_two = 3.5, 0
            done = True
        elif if_eat_two:
            reward_one, reward_two = 0, 3.5
            done = True
        else:
            reward_one, reward_two = 0, 0
            done = self.current_time >= self.game_length
        # import pdb; pdb.set_trace()
        '''
        logger.error(
            '(Action {}) Agent_one_pos: {}, (Action {}) Agent_two_pos: {}'.format(
                player_one_action, self.agent_one_pos,
                player_two_action, self.agent_two_pos
            )
        )
        logger.error('After')
        self.render()
        if done:
            logger.error('Done')
        # print self.agent_one_pos
        # import pdb; pdb.set_trace()
        '''

        return [[player_one_ob, reward_one, done],
                [player_two_ob, reward_two, done]]

    def if_eat(self, agent_pos, target_pos):
        if_eat = agent_pos[0] == target_pos[0] and agent_pos[1] == target_pos[1]
        return if_eat

    def render(self):
        matrix = np.zeros([self.maze_size, self.maze_size], dtype=np.int)
        matrix[self.agent_one_pos[0], self.agent_one_pos[1]] = 1
        matrix[self.agent_two_pos[0], self.agent_two_pos[1]] = -1
        logger.error('Rendering:')
        for i in range(len(matrix)):
            logger.error(matrix[i])
