# evaluate_snake.py
import tensorflow as tf
import numpy as np
import os
import time
import pygame
import sys

# Enable eager execution
tf.compat.v1.enable_eager_execution()

from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from rl_environment import SnakeEnv

def evaluate_policy(policy, env, num_episodes=5):
    """Evaluate the policy for a given number of episodes."""
    total_reward = 0
    total_steps = 0
    
    for episode in range(1, num_episodes + 1):
        print(f"\nStarting Episode {episode}/{num_episodes}")
        episode_reward = 0
        episode_steps = 0
        
        # Reset environment
        time_step = env.reset()
        # Add batch dimension to observation
        current_observation = tf.expand_dims(time_step.observation, 0)
        time_step = ts.TimeStep(
            step_type=tf.expand_dims(time_step.step_type, 0),
            reward=tf.expand_dims(time_step.reward, 0),
            discount=tf.expand_dims(time_step.discount, 0),
            observation=current_observation
        )
        
        while not time_step.is_last()[0]:  # Check first element of batch
            # Get action from policy
            action_step = policy.action(time_step)
            action = action_step.action.numpy()[0]  # Get first action from batch
            
            # Take step in environment
            time_step = env.step(action)
            # Add batch dimension to observation
            current_observation = tf.expand_dims(time_step.observation, 0)
            time_step = ts.TimeStep(
                step_type=tf.expand_dims(time_step.step_type, 0),
                reward=tf.expand_dims(time_step.reward, 0),
                discount=tf.expand_dims(time_step.discount, 0),
                observation=current_observation
            )
            
            reward = time_step.reward.numpy()[0]  # Get first reward from batch
            episode_reward += reward
            episode_steps += 1
            
            # Render the game using the underlying game instance
            env._game._update_ui()
            pygame.time.delay(50)  # Add small delay for visualization
            
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
        total_reward += episode_reward
        total_steps += episode_steps
        
        print(f"Episode {episode} finished:")
        print(f"  Steps: {episode_steps}")
        print(f"  Reward: {episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    
    print(f"\nEvaluation completed over {num_episodes} episodes:")
    print(f"Average steps per episode: {avg_steps:.2f}")
    print(f"Average reward per episode: {avg_reward:.2f}")
    
    return avg_reward, avg_steps

def main():
    # Initialize environment
    env = SnakeEnv(render_during_step=True)
    
    # Load saved policy
    policy_dir = os.path.join(os.getcwd(), 'saved_policy')
    print(f"Loading policy from: {policy_dir}")
    
    if not os.path.exists(policy_dir):
        print(f"Error: Policy directory not found at {policy_dir}")
        print("Please train the agent first using train_snake.py")
        sys.exit(1)
    
    # Load the saved policy
    policy = tf.saved_model.load(policy_dir)
    print("Policy loaded.\n")
    
    # Evaluate policy
    evaluate_policy(policy, env)
    
    # Clean up
    pygame.quit()
    
if __name__ == "__main__":
    main() 