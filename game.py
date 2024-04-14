import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25) #font size and style

#direction enum
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y') #ponit tuple


WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 20 
SPEED = 100

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.time_records = []
        self.reset()


    def reset(self):
       
        self.direction = Direction.RIGHT #moving right

        #snake represenation
        self.head = Point(self.w/2, self.h/2) #snake head at the start of game
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y), 
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)] # body of the snake

        self.score = 0
        self.food = None
        self._place_food() #to place food at random point in the board
        self.iteration = 0
        self.timetaken = 0
        self.starttime =  pygame.time.get_ticks()
        
    def _place_food(self):
        #place food at random point
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action) # perform action
        self.snake.insert(0, self.head) # update the head

        reward = 0
        game_over = False
        if self.is_collision() or self.iteration > 100*len(self.snake): # collision or snake doesn't improve performance
            game_over = True
            reward += -10
            self.timetaken = pygame.time.get_ticks() - self.starttime
            self.time_records.append(self.timetaken)
            if (len(self.time_records) >= 2):
                if self.time_records[-1] > self.time_records[-2]:  # snake survives for a longer time than previous game
                    reward += 1

            return reward, game_over, self.score, self.timetaken

        if self.head == self.food: # food is obtained
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score,self.timetaken


    def is_collision(self, pt=None):
        
        if pt is None: #if no point is explicitly specified
            pt = self.head # point is head of the snake
            
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: #if point is outside of the bounds
            return True
        
        if pt in self.snake[1:]: #if point is within the body of the snake
            return True
        return False


    def _update_ui(self):
        
        self.display.fill(BLACK)
        
        for i, pt in enumerate(self.snake): # Iterates through each point in the snake list
            if i == 0: # If the point is the head of the snake
                pygame.draw.rect(
                    self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE) # draw head
                )
                eye_center = (int(pt.x + BLOCK_SIZE // 3), int(pt.y + BLOCK_SIZE // 3))
                pygame.draw.circle(self.display, WHITE, eye_center, 2) # draw eye
            else:
                pygame.draw.rect(
                    self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE) # draw rest of the body
                )

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)) # draw food

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):# [straight, right, left]
       
        # determine the new direction and coordinates based on current direction and action to be taken
        direction_map = {
        Direction.RIGHT: {0: [Direction.RIGHT,(1,0)], 1: [Direction.DOWN,(0,1)], 2: [Direction.UP,(0,-1)]},
        Direction.DOWN: {0: [Direction.DOWN,(0,1)], 1: [Direction.LEFT,(-1,0)], 2: [Direction.RIGHT,(1,0)]},
        Direction.LEFT: {0: [Direction.LEFT,(-1,0)], 1: [Direction.UP,(0,-1)], 2: [Direction.DOWN,(0,1)]},
        Direction.UP: {0: [Direction.UP,(0,-1)], 1: [Direction.RIGHT,(1,0)], 2: [Direction.LEFT,(-1,0)]},
        } 
        
        key=0 # to indicate the action to perform
        if action == [0,1,0]: key = 1
        if action == [0,0,1]: key = 2
        new_dir=direction_map[self.direction][key]
        
        self.direction = new_dir[0]
        dx,dy = new_dir[1] 
        # calculate the new coordinates
        x = self.head.x + dx*BLOCK_SIZE
        y = self.head.y + dy*BLOCK_SIZE
        
        self.head = Point(x, y) #return the new postion of the head
