from copy import deepcopy
from typing import List, Tuple, Dict


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
*                                                       *
*         PUBLIC INTERFACES OF SEARCH MODULE            *
*                                                       *
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class Game:
    """  A game organizes a sequence of transitions in an environment. Each
    state in the environment has a node and, in between two nodes there is an action
    executed an a reward received this class stores """
    def __init__(self,observation_shape=None,action_size=None,num_players=None):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.num_players = num_players
        self.observations = []
        self.nodes = []
        self.actions = []
        self.masks = []
        self.players = []###
        self.rewards = []
        self.dones = []
        self.infos = [] 
        self.info = {} # don't confuse this attribute with the above. The above is a list
                       # of the infos returned by the environment. This one is for 
                       # any extra information we want to associate with this class instance

    def __len__(self):  #num of transitions
        return len(self.actions)

    def get_observations(self) -> List: #deprecated
        assert False
        return [n.get_observation() for n in self.nodes]

    def get_players(self) -> List: #deprecated
        return [n.get_player() for n in self.nodes]

    def get_dones(self) -> List: #this is quite the useless method
        return [n.is_terminal() for n in self.nodes]

    def get_player_rewards(self):
        rewards = {}
        assert len(self.players)-1 == len(self.rewards)
        for p,r in zip(self.players,self.rewards):
            if p not in rewards:
                rewards[p] = r
            else:
                rewards[p] += r
        return rewards




    

