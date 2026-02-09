import sys
import os
import argparse
import numpy as np
from collections import deque

# Add parent directory to path so we can import agent.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent import Agent
from game_no_ui import GameTraining


# Hyperparameters

# number of episodes to train for
episodes = 10000

# max number of steps per episode
max_steps = 200000

# initial/final epsilon for exploration-exploitation trade-off
epsilon_start = 1.0
epsilon_end = 0.001

# decay rate for epsilon
epsilon_decay = 0.9995

# rate at which the agent updates its weights
learning_rate = 0.001

# number of samples used in each training step
minibatch_size = 100

# discount factor for future rewards
gamma = 0.95

# maximum capacity of the replay memory
replay_memory_capacity = int(1e5)

# factor by which the target network is updated
interpolation_steps = 1e-2

# number of input features (bool of length 16)
input_size = 16

# number of possible actions
output_size = 4

# scores of the last 100 episodes
scores_of_episodes = deque(maxlen=100)


def parse_arguments():
    """Parse the arguments from the command line."""

    def grid_size_type(value):
        """Argument parser type for grid size."""
        grid = int(value)
        if grid < 4:
            raise argparse.ArgumentTypeError(
                "grid_size must be at least 4 (4x4 playable grid)."
            )
        return grid

    parser = argparse.ArgumentParser(
        description="Snake Game Training - No UI"
    )
    parser.add_argument(
        "grid_size",
        type=grid_size_type,
        nargs="?",
        default=10,
    )
    return parser.parse_args()


def train():
    try:
        args = parse_arguments()
        game = GameTraining(args.grid_size)

        # Change working directory to src/ so model paths resolve correctly
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))

        agent = Agent(
            input_size,
            output_size,
            learning_rate,
            replay_memory_capacity,
            interpolation_steps
        )
        agent.load_model()
        max_score = 0

        epsilon = epsilon_start
        if agent.epsilon != -1:
            epsilon = agent.epsilon
            max_score = max(max_score, agent.recorded_scores)

        for episode in range(0, episodes):
            game.reset()
            score = 0
            for t in range(max_steps):
                state_old = agent.get_state(game)
                action = agent.get_action(state_old, epsilon)
                move = [0, 0, 0, 0]
                move[action] = 1
                reward, done, score = game.run(move)
                if done:
                    state_new = state_old
                else:
                    state_new = agent.get_state(game)
                agent.step(
                    state_old,
                    action,
                    reward,
                    state_new,
                    done,
                    minibatch_size,
                    gamma,
                    interpolation_steps
                )
                if done:
                    break

            max_score = max(max_score, score)
            scores_of_episodes.append(score)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            agent.save_model(f"model_{episode}.pth")
            agent.save_data(max_score, epsilon)
            if episode % 50 == 0:
                print(
                    f"Episode: {episode}, "
                    f"Max Score: {max_score}, "
                    f"Current Score: {score}, "
                    f"Avg Score: {np.mean(scores_of_episodes):.2f}"
                )

    except SystemExit:
        return
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        return


if __name__ == "__main__":
    train()
