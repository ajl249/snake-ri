# train_snake.py
import tensorflow as tf
import numpy as np
import os
import time # Import the time module

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# Import the custom environment
from rl_environment import SnakeEnv

# Hyperparameters
num_iterations = 20000  # Total training iterations
initial_collect_steps = 1000  # Steps to collect before training starts
collect_steps_per_iteration = 1   # Steps collected in each training iteration
replay_buffer_max_length = 100000 # Max transitions stored in buffer

batch_size = 64  # Batch size for sampling from buffer
learning_rate = 1e-4 # Try this
log_interval = 200  # Print loss every N iterations
eval_interval = 1000 # Evaluate agent every N iterations (optional, can add later)
num_eval_episodes = 10 # Episodes to run for evaluation (optional)

# Define directory to save the policy
policy_dir = os.path.join(os.getcwd(), 'saved_policy')
if not os.path.exists(policy_dir):
    os.makedirs(policy_dir)

# --- Environment ---
# Instantiate the Python environment
train_py_env = SnakeEnv()
# eval_py_env = SnakeEnv() # Optional: Separate env for evaluation

# Wrap in TF environment
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env) # Optional

print('Observation Spec:')
print(train_env.time_step_spec().observation)
print('Action Spec:')
print(train_env.action_spec())

# --- Agent ---
# Define Q-Network
fc_layer_params = (128, 64) # Hidden layers: Dense 128 -> Dense 64

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# Define Optimizer using tf.keras.optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define DQN Agent
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss, # Standard DQN loss
    train_step_counter=train_step_counter)

agent.initialize()

# --- Replay Buffer ---
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size, # batch_size = 1 for the environment
    max_length=replay_buffer_max_length)

print("Agent Collect Data Spec:")
print(agent.collect_data_spec)
print(agent.collect_data_spec._fields)


# --- Data Collection ---
# Define a function to add experience to the buffer
def add_to_replay_buffer(experience):
    replay_buffer.add_batch(experience)

# Use a driver to handle interaction between agent and environment
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[add_to_replay_buffer],
    num_steps=collect_steps_per_iteration)

# Initial data collection phase (populate buffer before training)
print(f"Collecting initial {initial_collect_steps} steps...")

# Reset the environment to get the initial time_step
time_step = train_env.reset()

# Loop to collect the initial steps
for _ in range(initial_collect_steps):
    time_step, _ = collect_driver.run(time_step) # Pass current time_step

print("Initial data collection complete.")

# --- Dataset ---
# Create dataset pipeline for training
# Dataset generates trajectories with shape [Bx2x...] where B=batch_size, 2=num_steps
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, # Performance optimization
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3) # 2 steps for N-step TD learning (standard DQN uses 1 step usually, but 2 is common)

# Convert the dataset elements to trajectories for the agent
iterator = iter(dataset)
print("Dataset element spec:")
print(dataset.element_spec)

# --- Training Loop ---
# Optimize agent.train by wrapping it in tf.function.
# agent.train = common.function(agent.train) # Commented out to force eager execution

# Reset the train step
agent.train_step_counter.assign(0)

# Reset the environment
time_step = train_env.reset()

print(f"Starting training for {num_iterations} iterations...")
start_time = time.time() # Record start time

for i in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step) # Run one step

    # Sample a batch of data from the buffer and update the agent's network.
    # Check if buffer has enough data for a batch
    if replay_buffer.num_frames() > batch_size * 2: # Ensure enough steps for num_steps=2
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
    else: # Skip training if buffer isn't full enough yet
        train_loss = None

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        if train_loss is not None:
             print(f'step = {step}, loss = {train_loss}')
        else:
             print(f'step = {step}, loss = (Skipped - Replay buffer not full enough)')

    # Optional: Evaluation step
    # if step % eval_interval == 0:
    #     avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    #     print(f'step = {step}, Average Return = {avg_return}')
    #     returns.append(avg_return)

end_time = time.time() # Record end time
training_duration = end_time - start_time
print(f"\nTotal training time: {training_duration:.2f} seconds")

print("Training finished.")

# --- Save the Policy ---
print(f"Saving policy to: {policy_dir}")
tf_saver = policy_saver.PolicySaver(agent.policy)
tf_saver.save(policy_dir)
print("Policy saved.")

# Optional: Close the environment (Pygame window)
# train_env.close()
# eval_env.close()

# Optional: Visualization of returns (if evaluation was added)
# import matplotlib.pyplot as plt
# steps = range(0, num_iterations + 1, eval_interval)
# plt.plot(steps, returns)
# plt.ylabel('Average Return')
# plt.xlabel('Step')
# plt.ylim(bottom=0) # Adjust based on expected rewards
# plt.show() 