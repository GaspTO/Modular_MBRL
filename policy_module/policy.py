from game import Game
from node_module.node import Node 
from typing import List, Tuple, Dict

class Policy:
    def __init__(self,environment):
        self.environment = environment 

    def play_game(self) -> Game:
        """ override this method if necessary, but this one should be good enough 
            for most applications. It iterates through the environment, using the planning
            the choose a decision. At each step, it fills the game and node with the appropriate information
        """
        raise NotImplementedError


