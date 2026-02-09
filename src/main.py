from game import Game
from player import Player
from agent import Agent
import argparse


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

    parser = argparse.ArgumentParser(description="Snake Game - Learn2Slither")
    parser.add_argument(
        "grid_size",
        type=grid_size_type,
        nargs="?",
        default=10,
    )
    parser.add_argument(
        "-player",
        action="store_true",
        help="Run the game without AI.",
    )
    parser.add_argument(
        "-model",
        type=str,
        default="model_9999.pth",
        help="Model file to load (default: model_9999.pth).",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        if args.player:
            game = Player(args.grid_size)
            game.run()
        else:
            game = Game(args.grid_size)
            agent = Agent(
                input_size=16,
                output_size=4,
                learning_rate=0.001,
                replay_memory_capacity=int(1e5),
                interpolation_steps=1e-2,
            )
            agent.load_model(args.model)
            # Use the trained model (no exploration)
            epsilon = 0.0

            running = True
            while running:
                if not game.game_over:
                    state = agent.get_state(game)
                    action = agent.get_action(state, epsilon)
                    move = [0, 0, 0, 0]
                    move[action] = 1
                    reward, done, score = game.run(move)
                    if done:
                        game.game_over_text()
                        import pygame
                        pygame.display.update()
                else:
                    # Handle events while game over (restart / quit)
                    import pygame
                    from pygame import KEYDOWN, K_ESCAPE, K_RETURN
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == KEYDOWN:
                            if event.key == K_ESCAPE:
                                running = False
                            elif event.key == K_RETURN:
                                game.reset()

    except SystemExit:
        return
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        return


if __name__ == "__main__":
    main()
