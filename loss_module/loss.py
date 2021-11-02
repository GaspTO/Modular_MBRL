import torch
import numpy as np
from typing import Union, Optional, List, Tuple


'''
This is a simple common interface.

The main method is get_loss that returns a loss tensor and a dictionary with any specific
values that might be specific to each class
'''

class Loss:
    def __init__(self):
        self.device = torch.cuda.is_available()

    def get_loss(self,nodes:list,info={}) -> Tuple[torch.tensor,dict]:
        ''' 
            returns a loss for all nodes and a convenient dictionary with
            any relevant information 
        '''
        raise NotImplementedError



