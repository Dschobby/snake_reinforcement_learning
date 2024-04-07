from config import *
import pygame
from pygame.locals import *
import numpy as np


"""
Class controlling gameobject Snake
"""
class Snake:
    def __init__(self):
        self.pos = [round(WIDTH/2)*BLOCK_SIZE, round(HEIGHT/2)*BLOCK_SIZE]
        self.direction = 0
        self.snake_elements = [[self.pos.copy()[0] - BLOCK_SIZE, self.pos.copy()[1]],
                                self.pos.copy()]

    def draw(self, screen):
        for i in range(len(self.snake_elements)):
            pygame.draw.rect(screen, BLUE1, pygame.Rect(self.snake_elements[i][0], self.snake_elements[i][1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(screen, BLUE2, pygame.Rect(self.snake_elements[i][0]+round(BLOCK_SIZE*1/5), self.snake_elements[i][1]+round(BLOCK_SIZE*1/5), round(BLOCK_SIZE*3/5), round(BLOCK_SIZE*3/5)))

"""
Class controlling gameobject Food
"""
class Food:
    def __init__(self):
        self.pos = [(np.random.randint(-1,2)*round(WIDTH/4)+WIDTH/2)*BLOCK_SIZE, (np.random.randint(-1,2)*round(HEIGHT/4)+HEIGHT/2)*BLOCK_SIZE]

    def draw(self,screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.pos[0], self.pos[1], BLOCK_SIZE, BLOCK_SIZE))