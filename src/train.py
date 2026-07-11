import argparse
import numpy as np
from agent import Agent
from collections import deque
from game import Game


# Hyperparameters explanations

# max number of steps per episode
max_steps = 200000

# initial/final epsilon for exploration-exploitation trade-off
# exploration = agent is randomly choosing actions
# exploitation = agent is choosing the action with the highest Q-value
epsilon_start = 1.0
epsilon_end = 0.001

# decay rate for epsilon
epsilon_decay = 0.97

# rate at which the agent updates its weights
learning_rate = 0.01

# number of samples used in each training step
minibatch_size = 100

# discount factor for future rewards
gamma = 0.95

# maximum capacity of the replay memory
replay_memory_capacity = int(1e5)

# steps to interpolate target and online network
# factor by which the target network is updated
# 1 = target network is updated with the local network
# 0.05 = target network is updated with 5% of the local network
interpolation_steps = 5e-2

# number of input features
# 20 features:
# - move direction (LEFT, RIGHT, UP, DOWN) [4 bits]
# - 4 rays (left, right, up, down) with normalized distances [16 values]
#   each ray contains: [wall_dist, body_dist, green_dist, red_dist]
input_size = 20

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

    def sessions_type(value):
        """Argument parser type for sessions."""
        sessions = int(value)
        if sessions < 1:
            raise argparse.ArgumentTypeError(
                "sessions must be at least 1."
            )
        return sessions

    parser = argparse.ArgumentParser(description="Snake Game - Learn2Slither")
    parser.add_argument(
        "grid_size",
        type=grid_size_type,
        nargs="?",
        default=10,
    )
    parser.add_argument(
        "-sessions",
        type=sessions_type,
        default=1,
        help="Number of training/evaluation sessions (default: 1)",
    )
    parser.add_argument(
        "-save",
        type=str,
        default=None,
        help="Path to save the trained model (e.g., models/10sess.pth)",
    )
    parser.add_argument(
        "-load",
        type=str,
        default=None,
        help="Path to load a trained model (e.g., models/100sess.pth)",
    )
    parser.add_argument(
        "-visual",
        action="store_true",
        help="Enable or disable the graphical UI (default: on)",
    )
    parser.add_argument(
        "-eval",
        action="store_true",
        help="Run without learning (evaluation mode)",
    )
    parser.add_argument(
        "-step",
        action="store_true",
        dest="step",
        help="Step-by-step mode (press Space/Enter to advance each move)",
    )
    parser.add_argument(
        "-state",
        action="store_true",
        help="Display the snake state in the terminal after each move",
    )
    parser.add_argument(
        "-player",
        action="store_true",
        help="Run the game in human player mode",
    )
    return parser.parse_args()


def train():
    try:
        args = parse_arguments()
        no_ui = not args.visual
        game = Game(args.grid_size, no_ui=no_ui, show_state=False)
        agent = Agent(
            input_size,
            output_size,
            learning_rate,
            replay_memory_capacity,
            interpolation_steps
        )
        if args.load:
            agent.load_model(args.load)
        max_score = 0

        epsilon = epsilon_start
        if agent.epsilon != -1:
            epsilon = agent.epsilon

        for session in range(1, args.sessions + 1):
            game.reset()
            score = 0
            for t in range(max_steps):
                state_old = agent.get_state(game)
                action = agent.get_action(state_old, epsilon)
                move = [0, 0, 0, 0]
                move[action] = 1
                reward, done, score = game.play_step(move)
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
            agent.epsilon = epsilon
            agent.recorded_scores = max_score

            print(
                f"Session: {session}/{args.sessions}, "
                f"Score: {score}, "
                f"Avg Score: {np.mean(scores_of_episodes):.2f}"
                f"Max Score: {max_score}, "
            )

        if args.save:
            agent.save_model(args.save)
            print(f"Save learning state in {args.save}")

    except SystemExit:
        return
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        try:
            if args.save:
                agent.save_model(args.save)
                print(f"Save learning state in {args.save}")
        except Exception:
            pass
        return


if __name__ == "__main__":
    train()
