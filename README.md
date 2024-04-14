# SnakeGameAI

## Overview
This project implements the classic Snake game using Pygame and integrates reinforcement learning (RL) techniques for controlling the movement of the snake. The snake agent learns to navigate the game environment to collect food items while avoiding collisions with itself and the boundaries.

## Features
- game.py: Contains the implementation of the Snake game using Pygame, including game mechanics and user interface.
- agent.py: Defines the Snake agent class responsible for controlling the movement of the snake using reinforcement learning techniques.
- model.py: Defines the neural network model used by the agent to make decisions based on the game state.
- learner.py: Implements the reinforcement learning algorithm used by the agent to train and update its neural network model.
- dqn.pth: A pre-trained model file containing the weights of the neural network trained using the Deep Q-Network (DQN) algorithm.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Kanaga2004/SnakeGameAI

2. Navigate to the project directory:
  ```bash
  cd SnakeGameAI

3. Setup Environment (if conda environment is not set)
   - Install Pytorch
     ```bash
     pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

   - Install Pygame
     ```bash
     pip install pygame

 
