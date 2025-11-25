from game import Game
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
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        game = Game(args.grid_size)
        game.run()

    except SystemExit:
        return
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        return


if __name__ == "__main__":
    main()
