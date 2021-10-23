import random
import numpy as np


'''
*   This is the simplest replay buffer.
*   It appends the game to the end of a list
*   and uniformly samples a game based on the size of the trajectory
*   
*   This is the original reasoning of the replay buffer in https://www.nature.com/articles/nature14236
'''
class UniformBuffer():
    def __init__(self,max_buffer_size):
        self.buffer = []
        self.trainer = None
        self.max_buffer_size = max_buffer_size
        self.num = 0
    
    def add(self,nodes:list):
        assert isinstance(nodes,list), "Insert a list of nodes to be added"
        self.buffer.extend(nodes)
        self.num += len(nodes)
        while len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.max_buffer_size

    def sample(self, num=1):
        nodes = np.random.choice(self.buffer, num)
        return list(nodes)

    def get_size(self):
        return len(self.buffer)

    def get_num_of_added_samples(self):
        return self.num
