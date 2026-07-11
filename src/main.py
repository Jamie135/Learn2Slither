import argparse
import numpy as np
from collections import deque
from game import Game
from player import Player
from agent import Agent


# Hyperparameters
max_steps = 200000
epsilon_start = 1.0
epsilon_end = 0.001
epsilon_decay = 0.97
learning_rate = 0.001
minibatch_size = 100
gamma = 0.95
replay_memory_capacity = int(1e5)
interpolation_steps = 1e-2
input_size = 20
output_size = 4


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


def run_evaluation(args, game, agent):
    """Run evaluation sessions (no learning)."""
    epsilon = 0.0
    no_ui = not args.visual
    step = args.step

    for session in range(1, args.sessions + 1):
        game.reset()
        max_length = game.snake.length
        for t in range(max_steps):
            state = agent.get_state(game)
            action = agent.get_action(state, epsilon)
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.play_step(
                move, step=step
            )
            max_length = max(max_length, game.snake.length)
            if done:
                break

        print(
            f"Game over, score = {max_length - 3}, "
        )


def run_training(args, game, agent):
    """Run training sessions."""
    no_ui = not args.visual
    step = args.step
    scores = deque(maxlen=100)

    epsilon = epsilon_start
    if agent.epsilon != -1:
        epsilon = agent.epsilon
    max_score = 0

    for session in range(1, args.sessions + 1):
        game.reset()
        score = 0
        for t in range(max_steps):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, epsilon)
            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.play_step(
                move, step=step
            )
            if done:
                state_new = state_old
            else:
                state_new = agent.get_state(game)
            agent.step(
                state_old, action, reward, state_new,
                done, minibatch_size, gamma, interpolation_steps
            )
            if done:
                break

        max_score = max(max_score, score)
        scores.append(score)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent.epsilon = epsilon
        agent.recorded_scores = max_score

        print(
            f"Session: {session}/{args.sessions}, "
            f"Score: {score}, "
            f"Avg Score: {np.mean(scores):.2f}, "
            f"Max Score: {max_score}, "
        )

    if args.save:
        agent.save_model(args.save)
        print(f"Save learning state in {args.save}")


def main():
    try:
        args = parse_arguments()

        if args.player:
            game = Player(args.grid_size)
            game.run()
            return

        # AI mode
        no_ui = not args.visual
        game = Game(args.grid_size, no_ui=no_ui, show_state=args.state)
        agent = Agent(
            input_size=input_size,
            output_size=output_size,
            learning_rate=learning_rate,
            replay_memory_capacity=replay_memory_capacity,
            interpolation_steps=interpolation_steps,
        )

        if args.load:
            agent.load_model(args.load)

        if args.eval:
            run_evaluation(args, game, agent)
        else:
            run_training(args, game, agent)

    except SystemExit:
        return
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        try:
            if not args.player and args.save and not args.eval:
                agent.save_model(args.save)
                print(f"Save learning state in {args.save}")
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()
