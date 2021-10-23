import numpy as np
import torch 
from game import Game

class Node:
    def __init__(self):
        """ Don't use the attributes directly (except self.info), since subclasses might use their own and this interface is used everywhere
            throughout the architecture, so we're following more strict OOP practices.
            For e.g., in the monte-carlo tree search algorithm, this class is redifined so that get_value() returns
            an average value and not self._value. 
            
            Also, don't use batch dimension in the tensors you put here"""
        self._game:Game = None 
        self._idx_at_game:int = None
        self._action_mask:torch.Tensor = None
        self._player:int = None
        self._parent:Node = None #! make sure this triple is inserted
        self._parent_action:int = None
        self._parent_reward:float = None
        self._depth = None 
        self._value:float = None #updatable value
        self._encoded_state:torch.Tensor = None
        self._children:dict = {} #key is action:int, value is node
        self.info:dict = {}
            
    def _to_tensor(self,x):
        if isinstance(x,np.ndarray):
            return torch.tensor(x).float()
        elif isinstance(x,torch.Tensor):
            return x.float()
        elif x is None:
            return x
        else:
            raise ValueError("Can't convert input to torch Tensor")

    def detach_from_tree(self):
        self._children = {}
        self._parent = None

    ''' Game '''
    def get_game(self):
        return self._game

    def get_idx_at_game(self):
        return self._idx_at_game

    def set_game(self,game,idx):
        self._game = game
        self._idx_at_game = idx
        return self

    '''  Actions '''
    def get_action_mask(self):
        return self._to_tensor(self._action_mask)

    def set_action_mask(self,action_mask):
        self._action_mask = self._to_tensor(action_mask)
        return self

    def get_legal_actions(self,threshold=1):
        return torch.where(self.get_action_mask() >= threshold)[0]

    def get_illegal_actions(self,threshold=0):
        return torch.where(self.get_action_mask() <= threshold)[0]

    ''' Parent '''
    def get_parent(self):
        return self._parent

    def get_parent_action(self):
        return self._parent_action

    def get_parent_reward(self):
        return self._parent_reward

    def get_depth(self):
        if self.get_parent() is None:
            return 0
        else:
            return self._depth

    def set_parent_info(self,parent_node,parent_action,parent_reward):
        self._parent = parent_node
        self._parent_action = parent_action
        self._parent_reward = parent_reward
        self._depth = self._parent.get_depth() + 1
        return self


    ''' Successors '''
    def get_children(self) -> dict:
        return self._children

    def get_num_of_children(self):
        return len(self._children)

    def get_children_nodes(self):
        ''' this does not guarantee order'''
        return list(self._children.values())

    def set_children(self,children:dict):
        self._children = children
        return self

    def get_child(self,action):
        return self._children[action]

    def add_child(self,action,node):
        self._children[action] = node

    def get_child_reward(self,action):
        return self._children[action].get_parent_reward()

    def is_leaf(self):
        return len(self._children) == 0


    ''' Node Properties '''    
    #evaluation
    def successor_value(self,action):
        ''' value of successor for parent'''
        successor = self.get_child(action)
        if self.get_player() != successor.get_player():
            return (successor.get_parent_reward() - successor.get_value()) 
        else: 
            return (successor.get_parent_reward() + successor.get_value()) 
            
    def get_value(self):
        return self._value

    def set_value(self,value):
        self._value = value
        return self

    #encoded state
    def get_encoded_state(self):
        return self._to_tensor(self._encoded_state)

    def set_encoded_state(self,encoded_state):
        self._encoded_state = self._to_tensor(encoded_state)
        return self

    #player
    def set_player(self,player):
        self._player = player
        return self

    def get_player(self):
        return self._player







