import random

BLOCK_SIZE = 40


class AppleTraining:
    def __init__(self, grid_size, occupied_positions, kind):
        """
        Initialise the apples for training (no UI).
        """
        self.grid_size = grid_size
        self.kind = kind
        # No image loading needed for training
        self.randomize_position(occupied_positions)

    @staticmethod
    def free_position(grid_size, occupied):
        """Return a random (x, y) inside the playable area not in occupied."""
        while True:
            x = BLOCK_SIZE * random.randint(1, grid_size - 2)
            y = BLOCK_SIZE * random.randint(1, grid_size - 2)
            if (x, y) not in occupied:
                return x, y

    @classmethod
    def spawn_apples(cls, grid_size, snake):
        """
        Spawn two green apples and one red apple in free cells.
        """
        # Initial occupied positions: all snake segments.
        occupied = {(snake.x[i], snake.y[i]) for i in range(snake.length)}

        green_apples = []
        for _ in range(2):
            apple = cls(grid_size, occupied, "green")
            green_apples.append(apple)
            occupied.add((apple.x, apple.y))

        red_apple = cls(grid_size, occupied, "red")
        return green_apples, red_apple

    def randomize_position(self, occupied_positions):
        """Randomize apples position avoiding all occupied positions."""
        self.x, self.y = self.free_position(self.grid_size, occupied_positions)
