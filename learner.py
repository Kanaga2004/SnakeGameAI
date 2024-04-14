import torch
import torch.nn as nn
import torch.optim as optim

class Learner:
    def __init__(self, model):
        self.lr = 0.001
        self.gamma = 0.9
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # add another dimension to the elements taken as a input to the model
        state = torch.unsqueeze(state, 0) 
        next_state = torch.unsqueeze(next_state, 0)
        
        qValue = self.model(state) # predict the qValue 
        target = qValue.clone() # copy the target
        Q_new = reward 
        if not done:
            Q_new = reward + self.gamma * torch.max(self.model(next_state)) # compute the qValue 

        target[0][torch.argmax(action).item()] = Q_new # update the target qValues
       
        
        self.optimizer.zero_grad()
        loss = self.criterion(target,qValue) #calculate the loss
        loss.backward()

        self.optimizer.step() # optimize
