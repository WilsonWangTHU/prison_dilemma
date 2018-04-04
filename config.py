import argparse

def get_config():
    # get the parameters
    parser = argparse.ArgumentParser(description='PrisonDilemma.')

    parser.add_argument("--hidden_size", type=int, default=36)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--num_iteration", type=int, default=1000)
    parser.add_argument("--episode_per_batch", type=int, default=20)
    
    # the game
    parser.add_argument("--game_length", type=int, default=6)
    parser.add_argument("--history_length", type=int, default=5)

    args = parser.parse_args()

    return args
