# train_snake.py
import tensorflow as tf
import numpy as np
import os
import time

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

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0] if hasattr(avg_return, 'numpy') else float(avg_return)

# Hyperparameters
num_iterations = 30000  # Train longer
initial_collect_steps = 1000
collect_steps_per_iteration = 4
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-4
gamma = 0.99
epsilon_greedy_start = 1.0
epsilon_greedy_end = 0.05
epsilon_decay_steps = num_iterations // 2
target_update_period = 800
gradient_clipping = 1.0

log_interval = 200
eval_interval = 2000
num_eval_episodes = 10

# Define directory to save the policy
policy_dir = os.path.join(os.getcwd(), 'saved_policy')
if not os.path.exists(policy_dir):
    os.makedirs(policy_dir)

# --- Environment ---
train_py_env = SnakeEnv(render_during_step=False)
eval_py_env = SnakeEnv(render_during_step=False)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

print('Observation Spec:')
print(train_env.time_step_spec().observation)
print('Action Spec:')
print(train_env.action_spec())

# --- Agent ---
fc_layer_params = (128, 64)

q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# Define Optimizer using tf.keras.optimizers
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define epsilon decay schedule
epsilon_decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=epsilon_greedy_start,
    decay_steps=epsilon_decay_steps,
    end_learning_rate=epsilon_greedy_end)

# Define DQN Agent
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    epsilon_greedy=lambda: epsilon_decay_fn(train_step_counter),
    target_update_period=target_update_period,
    gradient_clipping=gradient_clipping,
    train_step_counter=train_step_counter)

agent.initialize()

# --- Replay Buffer ---
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

print("Agent Collect Data Spec:")
print(agent.collect_data_spec)
print(agent.collect_data_spec._fields)

# --- Data Collection ---
def add_to_replay_buffer(experience):
    replay_buffer.add_batch(experience)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[add_to_replay_buffer],
    num_steps=collect_steps_per_iteration)

# Initial data collection phase
print(f"Collecting initial {initial_collect_steps} steps...")
time_step = train_env.reset()
for _ in range(initial_collect_steps):
    time_step, _ = collect_driver.run(time_step)
print("Initial data collection complete.")

# --- Dataset ---
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)
print("Dataset element spec:")
print(dataset.element_spec)

# --- Training Loop ---
# Enable tf.function for faster training
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

print(f"Starting training for {num_iterations} iterations...")
start_time = time.time()
returns = [] # Initialize list to store evaluation returns

# Reset the environment
time_step = train_env.reset()

for i in range(num_iterations):
    # Collect steps using collect_policy and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    if replay_buffer.num_frames() >= batch_size * 2:
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
    else:
        train_loss = None

    step = agent.train_step_counter.numpy()

    # Log training information
    if step % log_interval == 0:
        if train_loss is not None:
            print(f'step = {step}, loss = {train_loss.numpy():.4f}, epsilon = {agent._epsilon_greedy:.3f}')
        else:
            print(f'step = {step}, loss = (Skipped - Replay buffer not full enough)')

    # Evaluate the agent's policy periodically
    if step % eval_interval == 0 and step > 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(f'------------ EVALUATION ------------')
        print(f'step = {step}, Average Return = {avg_return:.2f}')
        print(f'------------------------------------')
        returns.append(avg_return)

end_time = time.time()
training_duration = end_time - start_time
print(f"\nTotal training time: {training_duration:.2f} seconds")

print("Training finished.")

# --- Save the Policy ---
print(f"Saving policy to: {policy_dir}")
tf_saver = policy_saver.PolicySaver(agent.policy)
tf_saver.save(policy_dir)
print("Policy saved.")

# Optional: Plot training results if matplotlib is available
try:
    import matplotlib.pyplot as plt
    steps = range(eval_interval, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.title('Agent Performance Over Training')
    plt.grid(True)
    plt.show()
except ImportError:
    print("\nMatplotlib not found. Skipping plot generation.") 