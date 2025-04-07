# Snake Game with Deep Q-Learning

This project implements a Snake game using Pygame and trains an AI agent to play it using Deep Q-Learning (DQN) with TensorFlow and TF-Agents.

## Project Structure

- `snake_game.py`: The core Snake game implementation using Pygame
- `rl_environment.py`: TF-Agents environment wrapper for the Snake game
- `train_snake.py`: DQN agent training script
- `evaluate_snake.py`: Script to visualize the trained agent playing
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the agent:
```bash
python train_snake.py
```

## Evaluation

To watch the trained agent play:
```bash
python evaluate_snake.py
```

## Project Details

The Snake agent uses a Deep Q-Network (DQN) with the following features:
- State space: Snake's current direction, danger zones, and food location
- Action space: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
- Reward structure: +10 for eating food, -10 for collision, -0.1 for each move
- Neural Network: Dense layers (128 -> 64 -> 4) with ReLU activation 