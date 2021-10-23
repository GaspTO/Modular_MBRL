import numpy as np
import torch 
from game import Game

from node_module.node import Node

class BestFirstNode(Node):
    def __init__(self):
        super().__init__()
        self._visits = 0

    ''' visits '''
    def get_visits(self):
        return self._visits
    
    def set_visits(self,visits):
        self._visits = visits
    
    def increment_visits(self,n=1):
        self._visits += n
    