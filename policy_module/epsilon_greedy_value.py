from policy_module.simple_policy import SimplePolicy
from typing import List, Tuple, Dict
from node_module.node import Node
import numpy as np
import random



class EpsilonGreedyValue(SimplePolicy):
    def __init__(self,environment,planning,epsilon,reduction='successors'):
        super().__init__(environment,planning,reduction=reduction)
        self.epsilon = epsilon

    def play_game(self):
        ''' simple inheritance '''
        return super().play_game()

    def play_move(self,observation,player,mask:np.ndarray) -> Tuple[Node,int]:
        ''' returns node of observation and best action of it'''
        if not isinstance(mask,np.ndarray):
            raise ValueError("Mask should be a numpy array")
        node = self.get_planning_algorithm().plan(observation,player,mask)
        legal_actions, = np.where(mask == 1)
        if random.random() >= self.get_epsilon():
            best_action = max(legal_actions,key=lambda a:node.action_value(a))
        else: 
            best_action = random.choice(legal_actions)

        return node,best_action

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self,epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return str(self.get_epsilon()) + "-epsilonValueGreedy"

