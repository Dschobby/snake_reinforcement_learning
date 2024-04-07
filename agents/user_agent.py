import pygame
from pygame.locals import *


class User_agent:
    def __init__(self):
        self.vision = 4

    def act(self, state, train):
        for event in pygame.event.get():
            if event.type == QUIT:
                return -1
            if event.type == KEYDOWN:
                if event.key == K_d or event.key == K_RIGHT:
                    return 0
            if event.type == KEYDOWN:
                if event.key == K_s or event.key == K_DOWN:
                    return 1
            if event.type == KEYDOWN:
                if event.key == K_a or event.key == K_LEFT:
                    return 2
            if event.type == KEYDOWN:
                if event.key == K_w or event.key == K_UP:
                    return 3
        return -10