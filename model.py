import torch
from torch import nn
import torch.nn.functional as F
import os
import torch.optim as optim

class DQN(nn.Module):    
    def __init__(self, input_size=11, hidden_size=128, output_size=3):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)

    def save(self): #save the model
        if not os.path.exists('./model'):
            os.makedirs('./model')
        torch.save(self.state_dict(), './model/dqn.pth')
    
    def load(self): #load the model
        if os.path.exists('./model/dqn.pth'):
            self.load_state_dict(torch.load('./model/dqn.pth'))  
        else:
            return

