#!/usr/bin/env python

import numpy as np

COOP = 0
DEFECT = 1


class prison_env(object):

    def __init__(self, args):
        self.history_length = args.history_length
        self.game_length = args.game_length
        assert self.history_length < self.game_length
        # CC: -1, CD: -3, DC: 0, DD: -2
        self.reward_matrix = [[-1, -3], [0, -2]]

    def reset(self):
        self.player_one_state = [0] * self.history_length
        self.player_two_state = [0] * self.history_length
        self.current_time = 0

        player_one_ob = np.array(self.player_one_state + self.player_two_state)
        player_two_ob = np.array(self.player_two_state + self.player_one_state)
        return [[player_one_ob, 0, False],
                [player_two_ob, 0, False]]

    def step(self, player_one_action, player_two_action):
        self.player_one_state.pop(0)
        self.player_one_state.append(player_one_action)

        self.player_two_state.pop(0)
        self.player_two_state.append(player_two_action)

        player_one_ob = np.array(self.player_one_state + self.player_two_state)
        player_two_ob = np.array(self.player_two_state + self.player_one_state)

        reward_one = self.reward_matrix[player_one_action][player_two_action]
        reward_two = self.reward_matrix[player_two_action][player_one_action]
        # import pdb; pdb.set_trace()

        self.current_time += 1
        done = self.current_time >= self.game_length

        return [[player_one_ob, reward_one, done],
                [player_two_ob, reward_two, done]]
