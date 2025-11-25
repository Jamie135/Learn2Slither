import pygame
from snake import Snake
from apple import GreenApple
from pygame import KEYDOWN, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RETURN


BLOCK_SIZE = 40


class Game:
    def __init__(self, grid_size):
        """Initialise the game."""
        self.playable_grid_size = grid_size
        # Total grid, including the surrounding wall layer
        self.grid_size = grid_size + 2
        screen_size = (
            self.grid_size * BLOCK_SIZE,
            self.grid_size * BLOCK_SIZE,
        )

        pygame.init()
        pygame.display.set_caption("Snake Game")
        self.SCREEN_UPDATE = pygame.USEREVENT
        pygame.time.set_timer(self.SCREEN_UPDATE, 300)
        self.surface = pygame.display.set_mode(screen_size)
        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.greenApple = GreenApple(self.surface, self.grid_size)
        self.snake.draw()

    def play(self):
        """Move the snake and draw the apple."""
        self.snake.move()
        self.greenApple.draw()

    def run(self):
        """Run the game."""
        running = True
        started = False
        while running:
            # Handle events
            for event in pygame.event.get():

                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        started = True
                    if event.key == K_UP:
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        self.snake.move_down()
                    if event.key == K_LEFT:
                        self.snake.move_left()
                    if event.key == K_RIGHT:
                        self.snake.move_right()

                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SCREEN_UPDATE:
                    if started:
                        self.play()
            # Update the display
            pygame.display.update()
