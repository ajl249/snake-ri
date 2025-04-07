# evaluate_snake.py
import tensorflow as tf
import pygame # Need pygame for visualization
import time
import os # Import os for path joining

from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_loader # To load the saved policy

# Import the custom Python environment (not the TF wrapper this time)
from rl_environment import SnakeEnv
import snake_game # Need access to game constants like SPEED

# --- Configuration ---
policy_dir = os.path.join(os.getcwd(), "saved_policy") # Directory where the policy was saved
num_episodes = 5 # Number of episodes to visualize
frame_delay = 0.05 # Delay between frames in seconds for visualization speed

# --- Load Environment ---
# We need the original Python environment to manually control rendering/stepping
eval_py_env = SnakeEnv()
# We still wrap it slightly to easily get TF specs if needed, but won't use it for stepping
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# --- Load Policy ---
print(f"Loading policy from: {policy_dir}")
saved_policy = tf.saved_model.load(policy_dir)
print("Policy loaded.")

# --- Evaluation Loop ---
for episode in range(num_episodes):
    print(f"\nStarting Episode {episode + 1}/{num_episodes}")
    time_step = eval_tf_env.reset() # Use TF env just for reset convenience
    episode_return = 0.0
    steps = 0

    # Get the initial Pygame display surface from the underlying game
    # display_surface = eval_py_env._game.display # Not directly needed

    while not time_step.is_last():
        # Get action from the loaded policy
        # The policy requires a TimeStep object with batch dimension
        action_step = saved_policy.action(time_step)
        action = action_step.action.numpy()[0] # Extract action (remove batch dim)

        # Step the *Python* environment manually using the underlying game
        # We need the raw state, reward, done from the game's step method
        state, reward, done = eval_py_env._game.step(action)
        episode_return += reward
        steps += 1

        # Update the Pygame display (this is done inside _game.step)
        # No need to call _update_ui() separately

        # Construct the next TimeStep manually for the policy's next input
        # Note: We need to wrap the state numpy array for the TimeStep
        # Convert state back to TF Tensor with batch dimension
        current_observation = tf.convert_to_tensor([state], dtype=tf.float32)

        if done:
            # Create a termination TimeStep
            time_step = ts.termination(current_observation, reward)
        else:
            # Create a transition TimeStep
            time_step = ts.transition(current_observation, reward=reward, discount=1.0)


        # Add a small delay for visualization
        time.sleep(frame_delay)

        # Optional: Handle Pygame events like closing the window during evaluation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Pygame window closed during evaluation.")
                pygame.quit()
                exit() # Exit the script if window is closed

    print(f"Episode finished. Return = {episode_return}, Steps = {steps}")
    # No need to access _game.score directly, episode_return reflects reward sum
    # print(f"Episode finished. Score: {eval_py_env._game.score}, Return = {episode_return}, Steps = {steps}")


print("\nEvaluation finished.")
pygame.quit() # Close the Pygame window cleanly at the end 