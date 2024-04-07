from config import *
import objects
import pygame
from pygame.locals import *
import random
import sys
import time
import numpy as np
import torch
import agents.user_agent, agents.random_agent, agents.dqn_agent, agents.dcqn_agent



#Agents
AGENTS = ["user_agent", "random_agent", "dqn_agent", "dcqn_agent"]



"""
Main game class which is running and controlling the game
"""
class Game:
    
    def __init__(self, agent_name, device):

        #Initialize agent
        if not agent_name in AGENTS: sys.exit("Agent not defined")
        if device != "cpu" and device != "cuda": sys.exit("Computing device not available")
        if agent_name == "user_agent": 
            self.agent = agents.user_agent.User_agent()
            print("Initialize game with: User_agent")
        if agent_name == "random_agent": 
            self.agent = agents.random_agent.Random_agent()
            print("Initialize game with: Random_agent")
        if agent_name == "dqn_agent": 
            self.agent = agents.dqn_agent.DQN_agent(device)
            print("Initialize game with: DQN_agent")
            print("Trainable parameters: {}".format(sum(p.numel() for p in vars(self.agent)["model"].parameters())))
        if agent_name == "dcqn_agent": 
            self.agent = agents.dcqn_agent.DCQN_agent(device)
            print("Initialize game with: DCQN_agent")
            print("Trainable parameters: {}".format(sum(p.numel() for p in vars(self.agent)["model"].parameters())))
        self.device = device

        #Game objects (Get initialized new every game played)
        self.snake = None
        self.food = None
        self.score = None
        self.turn = None

        #Training mode for agent
        self.train = False

    def init_game(self):

        #Initialize game objects
        self.snake = objects.Snake()
        self.food = objects.Food()
        self.score = 0
        self.turn = 0

    def snake_handling(self):

        if self.snake.direction == 0:
            self.snake.pos[0] += BLOCK_SIZE
        if self.snake.direction == 1:
            self.snake.pos[1] += BLOCK_SIZE
        if self.snake.direction == 2:
            self.snake.pos[0] -= BLOCK_SIZE
        if self.snake.direction == 3:
            self.snake.pos[1] -= BLOCK_SIZE

        self.snake.snake_elements.append(self.snake.pos.copy())
        if not self.snake.pos == self.food.pos:
            self.snake.snake_elements.pop(0)

    def food_handling(self):

        if self.snake.pos == self.food.pos:
            self.score += 1
            self.turn = 0
            self.food.pos = [np.random.randint(round((SCREEN_WIDHT - BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE,
                             np.random.randint(round((SCREEN_HEIGHT - BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE]
            while self.food.pos in self.snake.snake_elements:
                self.food.pos = [np.random.randint(round((SCREEN_WIDHT - BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE,
                                 np.random.randint(round((SCREEN_HEIGHT - BLOCK_SIZE)/BLOCK_SIZE))*BLOCK_SIZE]
            
            return True
        
        return False


    def collision(self, pos):
        #Check for border collision
        if pos[0] < 0 or pos[1] < 0 or pos[0] > SCREEN_WIDHT - BLOCK_SIZE or pos[1] > SCREEN_HEIGHT - BLOCK_SIZE:
            return True
        
        #Check for self collision
        if pos in self.snake.snake_elements[0:-1]:
            return True

        return False
    
    def game_state_image(self):

        r = vars(self.agent)["vision"]
        state = np.zeros((WIDTH,HEIGHT))

        elem = np.transpose(np.array(self.snake.snake_elements[0:-1]) / BLOCK_SIZE).astype(int)
        pos = np.array(self.snake.pos) / BLOCK_SIZE
        food_pos = np.array(self.food.pos) / BLOCK_SIZE
        food = food_pos - pos

        state[elem[1],elem[0]] = -1

        for _ in range(r+1): state = np.insert(state, 0, -1, axis=1)
        for _ in range(r+1): state = np.insert(state, len(state.T), -1, axis=1)
        for _ in range(r+1): state = np.insert(state, 0, -1, axis=0)
        for _ in range(r+1): state = np.insert(state, len(state), -1, axis=0)

        for _ in range(round(pos[0]+1)): state = np.delete(state,0,axis=1)
        for _ in range(round(WIDTH - pos[0])): state = np.delete(state,len(state.T)-1,axis=1)
        for _ in range(round(pos[1]+1)): state = np.delete(state,0,axis=0)
        for _ in range(round(HEIGHT - pos[1])): state = np.delete(state,len(state)-1,axis=0)

        if np.sqrt(food[0]**2 +food[1]**2) > r:
            food_pointer = np.array([food[0]/np.sqrt(food[0]**2+food[1]**2), food[1]/np.sqrt(food[0]**2+food[1]**2)])
            max_length = np.max(np.abs(food_pointer))
            food_pointer = food_pointer * r/max_length
        else:
            food_pointer = np.array([food[0], food[1]])
        state[r + round(food_pointer[1]), r + round(food_pointer[0])] = 1

        state[r,r] = 0

        return state.tolist()

    def game_state_features(self):

        state = [self.snake.direction == 0, self.snake.direction == 1, self.snake.direction == 2, self.snake.direction == 3,
                 self.food.pos[0] > self.snake.pos[0], self.food.pos[1] > self.snake.pos[1], self.food.pos[0] < self.snake.pos[0], self.food.pos[1] < self.snake.pos[1]]
        
        state.append(self.collision([self.snake.pos[0] + BLOCK_SIZE, self.snake.pos[1]]) and not self.snake.direction == 2)
        state.append(self.collision([self.snake.pos[0], self.snake.pos[1] + BLOCK_SIZE]) and not self.snake.direction == 3)
        state.append(self.collision([self.snake.pos[0] - BLOCK_SIZE, self.snake.pos[1]]) and not self.snake.direction == 0)
        state.append(self.collision([self.snake.pos[0], self.snake.pos[1] - BLOCK_SIZE]) and not self.snake.direction == 1)

        return [int(x) for x in state]

    def reward(self, food_captured, invalid_action):

        reward = 0

        #dist_new = abs(self.snake.pos[0]-self.food.pos[0])**2 + abs(self.snake.pos[1]-self.food.pos[1])**2
        #if self.snake.direction == 0: dist_old = abs(self.snake.pos[0] - BLOCK_SIZE - self.food.pos[0])**2 + abs(self.snake.pos[1] - self.food.pos[1])**2
        #if self.snake.direction == 1: dist_old = abs(self.snake.pos[0] - self.food.pos[0])**2 + abs(self.snake.pos[1] - BLOCK_SIZE - self.food.pos[1])**2
        #if self.snake.direction == 2: dist_old = abs(self.snake.pos[0] + BLOCK_SIZE - self.food.pos[0])**2 + abs(self.snake.pos[1] - self.food.pos[1])**2
        #if self.snake.direction == 3: dist_old = abs(self.snake.pos[0] - self.food.pos[0])**2 + abs(self.snake.pos[1] + BLOCK_SIZE - self.food.pos[1])**2
        #if dist_new - dist_old < 0:
        #    reward = 0.1
        
        if self.collision(self.snake.pos):
            reward = -1 # reward -10 for colliding
        if food_captured:
            reward = 1 # reward +10 for finding food
        if invalid_action:
            reward = -1

        return reward

    def main(self, draw): 

        #Initialize pygame screen if wanted
        if draw:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake')
            clock = pygame.time.Clock()
            #pygame.time.delay(10000)

        #Initialize game
        active_episode = True
        self.init_game()

        #Game loop
        while active_episode:

            if draw:
                clock.tick(SPEED)

                #Check for closing game window
                if not isinstance(self.agent, agents.user_agent.User_agent):
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            active_episode = False

            #Get and execute agent action
            invalid_action = False
            if isinstance(self.agent, agents.dqn_agent.DQN_agent): state = self.game_state_features()
            else: state = self.game_state_image()
            action = self.agent.act(state, self.train)
            if not (action == -10): 
                if not (action + 2 == self.snake.direction) and not (action - 2 == self.snake.direction): 
                    self.snake.direction = action # make sure snake connot run into itself nothing stays direction
                else:
                    invalid_action = True
            if action == -1: active_episode = False

            #Updating environment
            self.snake_handling()
            food_captured = self.food_handling()
            
            #Check for collisions
            if self.collision(self.snake.pos): active_episode = False

            #Give state to experience buffer if in training mode of dqn_agent
            if self.train: #and (np.random.rand() > 0.5 or self.reward(food_captured) != 0):
                vars(self.agent)["buffer"][0].append(state)
                if isinstance(self.agent, agents.dqn_agent.DQN_agent): vars(self.agent)["buffer"][1].append(self.game_state_features())
                else: vars(self.agent)["buffer"][1].append(self.game_state_image())
                vars(self.agent)["buffer"][2].append(self.reward(food_captured, invalid_action))
                if action == 0: vars(self.agent)["buffer"][3].append(torch.Tensor([0]))
                if action == 1: vars(self.agent)["buffer"][3].append(torch.Tensor([1]))
                if action == 2: vars(self.agent)["buffer"][3].append(torch.Tensor([2]))
                if action == 3: vars(self.agent)["buffer"][3].append(torch.Tensor([3]))
            self.turn += 1

            #Update screen
            if draw:
                screen.fill(BLACK)

                self.snake.draw(screen)
                self.food.draw(screen)

                pygame.display.flip()

            #Terminate episode after reaching score of 100 or after given amount of turns
            if self.score >= 100:
                active_episode = False
            if self.turn >= 1000:
                active_episode = False

        #Quit pygame window
        if draw:
            pygame.display.quit()
            pygame.quit()

        return self.score
    

    def train_agent(self, draw, episodes, batches, hyperparameter):
            
            #Training control parameters
            convergence = 0 #parameter controlling if convergence happened
            loss = 0
            mean_score = []
            time_start = time.time()

            #Print training initials
            print("Start training process of agent")
            if self.device == "cuda": print("Using {} device".format(self.device), ": ", torch.cuda.get_device_name(0))
            else: print("Using {} device".format(self.device))
            print("Used training hyperparameters: ",hyperparameter)

            #Check if agent is trainable
            if not isinstance(self.agent, agents.dqn_agent.DQN_agent) and not isinstance(self.agent, agents.dcqn_agent.DCQN_agent):
                sys.exit("Agent is not trainable")

            self.train = True

            for episode in range(1, episodes + 1):

                #print(len(vars(self.agent)["buffer"][0]))

                #Specify episode lr and epsilon
                eps = hyperparameter["eps_end"] + (hyperparameter["eps_start"] - hyperparameter["eps_end"]) * np.exp(-1. * episode /episodes * 10)
                lr = hyperparameter["lr_end"] + (hyperparameter["lr_start"] - hyperparameter["lr_end"]) * np.exp(-1. * episode /episodes * 10)
                vars(self.agent)["lr"] = lr
                vars(self.agent)["batch_size"] = hyperparameter["batch_size"]
                vars(self.agent)["gamma"] = hyperparameter["gamma"]
                vars(self.agent)["epsilon"] = eps

                #Run an episode
                _ = self.main(draw)
                
                #Train agent
                for i in range(batches):
                    loss += self.agent.train()

                #Test agent
                self.train = False
                test_score = self.main(False)
                mean_score.append(test_score)
                if test_score == 50: convergence += 1 #look if agent has perfectly performed the game
                else: convergence = 0
                
                #Print training perfomance log
                time_step = time.time()
                if episode % 10 == 0 or convergence == 2: 
                    print("Episode: [{}/{}]".format(episode, episodes) + 
                        "    -Time: [{}<{}]".format(time.strftime("%M:%S", time.gmtime(time_step-time_start)), time.strftime("%M:%S", time.gmtime((time_step-time_start) * episodes/episode))) +
                        " {}s/it".format(round((time_step-time_start)/episode,1)) +
                        "    -Loss: {}".format(round(loss/batches,6)) + 
                        "    -MeanTestScore: {}".format(round(np.mean(mean_score))))
                    mean_score = []
                loss = 0

                #Terminate training if agent never collides after two training procedures in a row
                if convergence == 2: 
                    print("Agent performed faultless")
                    break
                self.train = True


            self.train = False
            
            print("Training finished after {} episodes".format(episode))
