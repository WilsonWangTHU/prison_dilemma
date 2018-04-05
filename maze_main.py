#!/usr/bin/env python

import tensorflow as tf
from config import get_config
import agent
from maze_env import maze_env
import logger


def policy_rollout(env, agent_one, agent_two):
    '''
        @brief: run for one episode
    '''

    # logger.warning('Reset!?')
    player_info = env.reset()
    agent_one_infolist, agent_two_infolist = [], []
    done = player_info[0][2]
    old_act_one = None
    old_act_two = None

    while not done:
        act_one = agent_one.act(player_info[0][0],
                                [player_info[1][0], old_act_two])
        act_two = agent_two.act(player_info[1][0],
                                [player_info[0][0], old_act_one])
        old_act_one, old_act_two = act_one, act_two
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

    env = maze_env(args)
    args.input_size = 12
    args.num_actions = 4

    session = tf.Session()
    # TYPE SS:
    if args.exp_type == 'SS':
        agent_one = agent.selfish_agent(args, session, name_scope='Selfish_A')
        agent_two = agent.selfish_agent(args, session, name_scope='Selfish_B')
    elif args.exp_type == 'SN':
        agent_one = agent.naive_agent(args, session, name_scope='Naive_A')
        agent_two = agent.selfish_agent(args, session, name_scope='Selfish_B')
    elif args.exp_type == 'SP':
        agent_one = agent.punishment_agent(args, session, name_scope='Punish_A')
        agent_two = agent.selfish_agent(args, session, name_scope='Selfish_B')
    elif args.exp_type == 'AN':
        agent_one = agent.adaptive_agent(args, session, name_scope='Adaptive_A')
        agent_two = agent.naive_agent(args, session, name_scope='Naive_B')
    elif args.exp_type == 'AS':
        agent_one = agent.adaptive_agent(args, session, name_scope='Adaptive_A')
        agent_two = agent.selfish_agent(args, session, name_scope='Selfish_B')
    elif args.exp_type == 'AA':
        agent_one = agent.adaptive_agent(args, session, name_scope='Adaptive_A')
        agent_two = agent.adaptive_agent(args, session, name_scope='Adaptive_B')

    session.run(tf.global_variables_initializer())

    for i_iteration in range(args.num_iteration):

        logger.info('Current Iteration {}'.format(i_iteration))

        for _ in range(args.episode_per_batch):

            policy_rollout(env, agent_one, agent_two)

        agent_one.train_step()
        agent_two.train_step()

    agent_one.save_npy(args.task, agent_one._name_scope)
    agent_two.save_npy(args.task, agent_two._name_scope)


if __name__ == "__main__":
    main()
