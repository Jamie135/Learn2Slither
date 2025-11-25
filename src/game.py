import pygame
from snake import Snake
from apple import Apple
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
        pygame.time.set_timer(self.SCREEN_UPDATE, 400)
        self.surface = pygame.display.set_mode(screen_size)
        # Snake and apple receive the total grid size (including walls)
        self.snake = Snake(self.surface, self.grid_size)
        self.green_apples, self.red_apple = Apple.spawn_apples(
            self.surface,
            self.grid_size,
            self.snake,
        )
        self.snake.draw()

    def start_text(self):
        """Display a start message before the game begins."""
        font = pygame.font.SysFont(None, 48)
        text = "Press an arrow key to start"
        color = (255, 255, 255)
        shadow_color = (0, 0, 0)

        text_surface = font.render(text, True, color)
        shadow_surface = font.render(text, True, shadow_color)

        rect = text_surface.get_rect(
            center=(self.surface.get_width() // 2, self.surface.get_height() // 2),
        )

        # Draw a simple shadow for better readability
        shadow_rect = rect.move(2, 2)
        self.surface.blit(shadow_surface, shadow_rect)
        self.surface.blit(text_surface, rect)

    def play(self):
        """Move the snake and draw the apple."""
        self.snake.move()
        for apple in self.green_apples:
            apple.draw()
        self.red_apple.draw()

    def run(self):
        """Run the game."""
        running = True
        started = False
        while running:
            # Handle events
            for event in pygame.event.get():

                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        started = True
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        started = True
                        self.snake.move_down()
                    if event.key == K_LEFT:
                        started = True
                        self.snake.move_left()
                    if event.key == K_RIGHT:
                        started = True
                        self.snake.move_right()

                if event.type == pygame.QUIT:
                    running = False
                elif event.type == self.SCREEN_UPDATE:
                    if started:
                        self.play()
            # If the game hasn't started yet, overlay the start text.
            if not started:
                self.start_text()
            # Update the display
            pygame.display.update()
