# rl_environment.py
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# Import the game itself
from snake_game import SnakeGame, Direction, Point, BLOCK_SIZE

tf.compat.v1.enable_v2_behavior() # Make sure TF2 behavior is enabled

class SnakeEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self._game = SnakeGame() # Initialize the game internally

        # Define action spec: 4 discrete actions (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        # Define observation spec based on the state from SnakeGame.get_state()
        # State: [danger_straight, danger_right, danger_left,
        #         dir_l, dir_r, dir_u, dir_d,
        #         food_l, food_r, food_u, food_d] - 11 values
        initial_state = self._game.get_state()
        self._observation_spec = array_spec.ArraySpec(
            shape=initial_state.shape, dtype=np.float32, name='observation')

        self._episode_ended = False
        # Initialize state by resetting the game
        self._state = self._game.reset()


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """Resets the environment and returns the initial time step."""
        # print("Resetting environment")
        self._state = self._game.reset()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        """Applies the action and returns the next time step."""
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Convert action tensor/numpy to Python int if necessary
        if hasattr(action, 'numpy'):
             action_int = action.numpy()
        else:
             action_int = int(action) # Ensure it's a Python int for the game logic


        # Check if the action is valid (not immediate reversal)
        # This logic is now handled inside snake_game._move,
        # so we can directly pass the action.

        # Step the game
        self._state, reward, done = self._game.step(action_int)

        # Convert reward to float32
        reward = np.float32(reward)

        if done:
            self._episode_ended = True
            # print(f"Episode ended. Score: {self._game.score}, Reward: {reward}")
            return ts.termination(self._state, reward)
        else:
            # print(f"Step taken. Reward: {reward}")
            return ts.transition(self._state, reward=reward, discount=1.0)

# --- Example usage (Optional: For testing the PyEnvironment) ---
if __name__ == '__main__':
    # Need to import utils if running the example:
    from tf_agents.environments import utils

    environment = SnakeEnv()
    utils.validate_py_environment(environment, episodes=5)

    # Or manually test:
    # time_step = environment.reset()
    # print("Initial Time Step:")
    # print(time_step)
    # cumulative_reward = time_step.reward

    # for _ in range(10):
    #     action = tf.constant(np.random.randint(0, 4), dtype=np.int32) # Random action
    #     time_step = environment.step(action)
    #     print(f"\nAction: {action.numpy()}")
    #     print("Next Time Step:")
    #     print(time_step)
    #     cumulative_reward += time_step.reward

    # print("\nFinal Cumulative Reward:", cumulative_reward) 