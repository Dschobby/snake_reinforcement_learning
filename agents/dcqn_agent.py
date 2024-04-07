import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


"""
Convolutional Neural Network model structure for the DCQN agent
"""    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),

            nn.Flatten(),
        )
        self.linear = nn.Linear(128, 4)
        
    def forward(self, x):
        return self.linear(self.model(x))


"""
DCQN agent for reinforcement learning
"""
class DCQN_agent:
    def __init__(self, device):
        #Set up model
        self.device = device
        self.model = CNN().to(device)
        self.vision = 4

        #Set up experience buffer
        self.buffer = [[],[],[],[]]
        self.buffer_size = 10000

        #Training hyperparameters
        self.lr = 0.001
        self.batch_size = 64
        self.gamma = 0.2
        self.epsilon = 0.0

        #Set up training helpers
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state, train):
        state = torch.Tensor(state).to(self.device)

        #Choose action with max q value of if in training mode choose action according to epsilon greedy policy
        if np.random.rand() > self.epsilon or not train:
            with torch.no_grad():
                return self.model(state.reshape(1,1,len(state),len(state))).reshape(-1).data.max(0)[1].detach().cpu()
        else:
            return torch.randint(4,(1,))[0]

    def train(self):
        #Check buffer size
        if len(self.buffer[0]) < self.batch_size:
            return 0
        #Delete old data from buffer if buffer size is surpassed
        if len(self.buffer[0]) > self.buffer_size:
            for _ in range(len(self.buffer[0])-self.buffer_size):
                ind = np.random.randint(len(self.buffer[0]))
                del self.buffer[0][ind]
                del self.buffer[1][ind]
                del self.buffer[2][ind]
                del self.buffer[3][ind]
   
        #Create data batch
        batch_ind = torch.randperm(len(self.buffer[0]))[0:self.batch_size]
        batch_state = torch.Tensor(self.buffer[0])[batch_ind].to(self.device)
        batch_next_state = torch.Tensor(self.buffer[1])[batch_ind].to(self.device)
        batch_action = torch.Tensor([self.buffer[3]]).long().reshape(len(self.buffer[3]),-1)[batch_ind].to(self.device)
        batch_reward = torch.Tensor(self.buffer[2])[batch_ind].to(self.device)

        #Calculate expected and current q-value
        current_q_values = self.model(batch_state.reshape(self.batch_size,1,len(batch_state[0]),len(batch_state[0]))).gather(1,batch_action)
        with torch.no_grad():
            max_next_q_values = self.model(batch_next_state.reshape(self.batch_size,1,len(batch_state[0]),len(batch_state[0]))).detach().max(1)[0]
        expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        #loss
        loss = self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        #backpropagation of loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 10) # gradient clipping for preventing exploding gradients
        self.optimizer.step()

        return loss.item()
