# snake_game.py
import pygame
import random
import numpy as np
from collections import namedtuple

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Define Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Game Constants
BLOCK_SIZE = 20
SPEED = 100 # Adjusted later for RL agent speed if needed
WIDTH = 640
HEIGHT = 480

# Point tuple for coordinates
Point = namedtuple('Point', 'x, y')

# Directions (for internal logic, not RL actions yet)
class Direction:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class SnakeGame:
    def __init__(self, w=WIDTH, h=HEIGHT, render=True):
        self.w = w
        self.h = h
        self.render = render # Store the flag
        if self.render: # Only initialize display if rendering
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        else:
            self.display = None
            self.clock = None # No clock needed if not rendering/ticking
        self.reset()

    def reset(self):
        """Reset the game state."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.game_over = False
        # Return initial state for RL env reset
        return self.get_state()

    def _place_food(self):
        """Place food randomly, ensuring it doesn't overlap with the snake."""
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def _move(self, action):
        """
        Move the snake based on the action.
        Action mapping: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        Prevents immediate reversal.
        """
        current_direction = self.direction
        new_direction = current_direction

        if action == Direction.UP and current_direction != Direction.DOWN:
            new_direction = Direction.UP
        elif action == Direction.DOWN and current_direction != Direction.UP:
            new_direction = Direction.DOWN
        elif action == Direction.LEFT and current_direction != Direction.RIGHT:
            new_direction = Direction.LEFT
        elif action == Direction.RIGHT and current_direction != Direction.LEFT:
            new_direction = Direction.RIGHT
        # If action would cause reversal, continue in the current direction

        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
        self.snake.insert(0, self.head)

    def _is_collision(self, pt=None):
        """Check for collisions: walls or self."""
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """Draw the current game state only if rendering is enabled."""
        if not self.render: # Check the flag
            return

        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def get_state(self):
        """Return the current state as a NumPy array for RL."""
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger Right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger Left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < head.x,  # Food left
            self.food.x > head.x,  # Food right
            self.food.y < head.y,  # Food up
            self.food.y > head.y   # Food down
        ]
        # Return as float32 for TF-Agents compatibility
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Take an action, update the game, return (new_state, reward, done_flag).
        """
        self.frame_iteration += 1

        # 1. Handle quit event (only relevant if rendering)
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 2. Move snake
        self._move(action) # Action is 0, 1, 2, or 3

        # 3. Check for game over conditions
        reward = -0.01 # Small negative reward for each step
        self.game_over = False

        # Add timeout condition
        if self.frame_iteration > 150 * (len(self.snake) + 1):
            self.game_over = True
            reward = -1.0 # Negative reward for timeout
            current_state = self.get_state()
            return current_state, reward, self.game_over

        # Check collision
        if self._is_collision():
            self.game_over = True
            reward = -1.0 # Negative reward for collision
            current_state = self.get_state()
            return current_state, reward, self.game_over

        # 4. Check for food
        if self.head == self.food:
            self.score += 1
            reward = 1.0 # Positive reward for eating food
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI & Clock (only if rendering)
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. Return state, reward, done
        current_state = self.get_state()
        return current_state, reward, self.game_over

# --- Main execution block for human play ---
if __name__ == '__main__':
    game = SnakeGame()

    # Game loop for human play
    while True:
        # Default action is current direction (keep moving straight)
        action = game.direction

        # Get human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    action = Direction.LEFT
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    action = Direction.RIGHT
                elif event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    action = Direction.UP
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    action = Direction.DOWN
                # No need to break here, process the chosen action in step

        # Perform game step using the determined action
        _, reward, done = game.step(action)

        if done:
            # Could display final score and wait or restart
            print(f"Game Over! Final Score: {game.score}")
            # For now, just quit on game over in human mode
            pygame.quit()
            quit()

            # Or reset and continue:
            # game.reset()