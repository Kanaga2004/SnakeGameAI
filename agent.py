import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import DQN
from learner import Learner
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
BLOCK_SIZE = 20

class Agent:

    def __init__(self):
        self.gamecount = 0
        self.epsilon = 0 # randomness
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = DQN()
        self.trainer = Learner(self.model)


    def get_state(self, game):
     # [danger straight, danger right, danger left, moving r, moving l, moving u, moving d ,food r, food l, food u, food d]
        head = game.snake[0]
        
        #calculate the coordinates of taking the possible action
        pt_right = Point(head.x+BLOCK_SIZE,head.y)
        pt_left = Point(head.x-BLOCK_SIZE,head.y)
        pt_up = Point(head.x,head.y-BLOCK_SIZE)
        pt_down = Point(head.x,head.y+BLOCK_SIZE)
        
        #compute the danger in moving straight, right and left w.r.t to current position and direction
        danger_straight = int((game.direction == Direction.RIGHT  and game.is_collision(pt_right)) or
                              (game.direction == Direction.LEFT and game.is_collision(pt_left)) or 
                              (game.direction == Direction.DOWN and game.is_collision(pt_down)) or 
                              (game.direction == Direction.UP and game.is_collision(pt_up)))
        danger_right = int((game.direction == Direction.RIGHT  and game.is_collision(pt_down)) or 
                           (game.direction == Direction.LEFT and game.is_collision(pt_up)) or 
                           (game.direction == Direction.DOWN and game.is_collision(pt_left)) or 
                           (game.direction == Direction.UP and game.is_collision(pt_right)))
        danger_left = int((game.direction == Direction.RIGHT  and game.is_collision(pt_up)) or 
                          (game.direction == Direction.LEFT and game.is_collision(pt_down)) or 
                          (game.direction == Direction.DOWN and game.is_collision(pt_right)) or 
                          (game.direction == Direction.UP and game.is_collision(pt_left)))
        
        moving_r = int(game.direction == Direction.RIGHT)
        moving_l = int(game.direction == Direction.LEFT)
        moving_u = int(game.direction == Direction.UP)
        moving_d = int(game.direction == Direction.DOWN)
        
        #compute the food placement w.r.t to current position
        food_r = game.food.x < head.x
        food_l = game.food.x > head.x
        food_u = game.food.y < head.y
        food_d = game.food.y > head.y
        
        state = [danger_straight, danger_right, danger_left, moving_r, moving_l, moving_u, moving_d, food_r, food_l, food_u, food_d]

        return np.array(state, dtype=int)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #store in memory

    def train_memory(self, short_memory=False):
        
        if short_memory: #training each state per action
            state, action, reward, next_state, done = self.memory[-1]
            self.trainer.train_step(state, action, reward, next_state, done)
            
        else: # training 1000 random states per game
            if len(self.memory) > BATCH_SIZE:
                start_index = random.randint(0, len(self.memory) - BATCH_SIZE)
                end_index = start_index + BATCH_SIZE
                memory_to_sample = list(self.memory)[start_index:end_index]
            else:
                memory_to_sample = list(self.memory)
            for state, action, reward, next_state, done in memory_to_sample:
                self.trainer.train_step(state, action, reward, next_state, done)

       
    def get_action(self, state):
        #EXPLORATION: 
        self.epsilon = 80 - self.gamecount
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
        #EXPLOITATION: 
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            qValue = self.model(state_tensor)
            idx = torch.argmax(qValue).item()
        action = []
        for i in range(3):
            if i == idx:
                action.append(1)
            else:
                action.append(0)
        return action



if __name__ == '__main__':
    choice = input("Enter 'train' to train the model or 'play' to use the trained model: ")
    
    #TRAINING
    if choice.lower() == 'train':
        scores = []
        mean_scores = []
        time_survived = []
        std_dev_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = SnakeGame()
        while True:
            
            curr_state= agent.get_state(game) # get state
            action = agent.get_action(curr_state) # get action
            reward, gameover, score , timetaken = game.play_step(action) # play action
            new_state= agent.get_state(game) # get new state
            agent.store(curr_state, action, reward, new_state, gameover) # store the sample in memory
            agent.train_memory(True) # train short memory
            
            if gameover: # GAMEOVER
                game.reset()
                agent.gamecount += 1
                agent.train_memory() # train long memory
                
                if score > record: # best score
                    record = score
                    agent.model.save() # save model parameters
                
                print('Game', agent.gamecount, 'Score', score, 'Record:', record)

                scores.append(score)
                time_survived.append(timetaken)
                mean_score = total_score / agent.gamecount
                mean_scores.append(mean_score)
                total_score += score
                mean_scores.append(mean_score)
                std_dev_scores.append(np.std(scores))

                plot(scores, mean_scores, time_survived, std_dev_scores)
    
    #EVALUATION        
    elif choice.lower() == 'play':
        scores = []
        mean_scores = []
        time_survived = []
        std_dev_scores = []
        total_score = 0
        record = 0
        # Load the trained model
        agent = Agent()
        agent.model.load_state_dict(torch.load('model/dqn.pth'))
        agent.model.eval()  # Set the model to evaluation mode
        game = SnakeGame()
        count = 0
        while True and count <= 100:
            
            curr_state = agent.get_state(game) # get state
            state_tensor = torch.tensor(curr_state, dtype=torch.float) # convert to tensor
            qValue = agent.model(state_tensor) # predict qValue
            idx = torch.argmax(qValue).item() # index with max qValue
            action = []
            for i in range(3):
                if i == idx:
                    action.append(1)
                else:
                    action.append(0)
            reward, gameover, score, timetaken = game.play_step(action) # play optimal action
            if gameover: # GAMEOVER
                agent.gamecount += 1
                print('Game Number ',agent.gamecount,': Score:', score)
                game.reset()
                scores.append(score)
                time_survived.append(timetaken)
                mean_score = total_score / agent.gamecount
                mean_scores.append(mean_score)
                total_score += score
                mean_scores.append(mean_score)
                std_dev_score = np.std(scores)
                std_dev_scores.append(std_dev_score)
                plot(scores, mean_scores, time_survived, std_dev_scores)
                count+=1
        # Calculate additional evaluation metrics
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        win_rate = np.sum(np.array(scores) > 25) / count
        avg_time_survived = np.mean(time_survived)
        std_dev_score = np.std(scores)
        print("Evaluation Metrics:")
        print("Average Score:", avg_score)
        print("Maximum Score:", max_score)
        print("Minimum Score:", min_score)
        print("Win Rate:", win_rate)
        print("Average Time Survived:", avg_time_survived)
        print("Standard Deviation of Scores:", std_dev_score)
    else:
        print("Invalid choice. Please enter 'train' or 'play'.")