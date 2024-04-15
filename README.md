# SnakeGameAI

## Overview

The Snake Game AI project aims to implement an intelligent agent capable of playing the classic Snake game autonomously. Leveraging reinforcement learning techniques, the agent learns to navigate the game environment, avoid obstacles, and consume food items to increase its score. The project consists of several key components, including the game engine itself, the reinforcement learning agent, and the neural network model used for decision-making. Through iterative training sessions, the agent learns optimal strategies for maximizing its score while avoiding collisions with itself and the game boundaries.

## Features
- game.py: Contains the implementation of the Snake game using Pygame, including game mechanics and user interface.
- agent.py: Defines the Snake agent class responsible for controlling the movement of the snake using reinforcement learning techniques.
- model.py: Defines the neural network model used by the agent to make decisions based on the game state.
- learner.py: Implements the reinforcement learning algorithm used by the agent to train and update its neural network model.
- dqn.pth: A pre-trained model file containing the weights of the neural network trained using the Deep Q-Network (DQN) algorithm.
- helper.py: Defines the visualization plots 

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

   - Install Matplotlib, Numpy and Panda
     ```bash
     pip install matplotlib
     pip install numpy
     pip install panda

## Usage

To run the code, navigate to the project directory in the terminal and type the following command:
```bash
python agent.py
```

On running this code you will receive a prompt
```bash
Hello from the pygame community. https://www.pygame.org/contribute.html
Enter 'train' to train the model or 'play' to use the trained model:
```
Type train to train a new model and replace the existing model or type play to see how the trained model works.




 
