#!/usr/bin/env python

import tensorflow as tf
from config import get_config
import agent
from env import prison_env
import logger
import time


def policy_rollout(env, agent_one, agent_two):
    '''
        @brief: run for one episode
    '''

    player_info = env.reset()
    agent_one_infolist, agent_two_infolist = [], []
    done = player_info[0][2]

    while not done:
        act_one = agent_one.act(player_info[0][0])
        act_two = agent_two.act(player_info[1][0])
        done = player_info[0][2]

        agent_one_infolist.append(
            player_info[0] + [act_one] + player_info[1] + [act_two]
        )
        agent_two_infolist.append(
            player_info[1] + [act_two] + player_info[0] + [act_one]
        )
        player_info = env.step(act_one, act_two)

    agent_one.record_episode_info(agent_one_infolist)
    agent_two.record_episode_info(agent_two_infolist)


def main():
    args = get_config()

    env = prison_env(args)
    args.input_size = 2 * args.history_length
    args.num_actions = 2

    session = tf.Session()
    agent_one = agent.selfish_agent(args, session, name_scope='selfish_agent')
    agent_two = agent.selfish_agent(args, session, name_scope='naive_agent')

    session.run(tf.global_variables_initializer())

    for i_iteration in range(args.num_iteration):

        logger.info('Current Iteration {}'.format(i_iteration))

        for _ in range(args.episode_per_batch):

            policy_rollout(env, agent_one, agent_two)

        agent_one.train_step()
        agent_two.train_step()

    current_time = time.time()
    agent_one.save_npy(current_time, agent_one._name_scope)
    agent_two.save_npy(current_time, agent_two._name_scope)


if __name__ == "__main__":
    main()
