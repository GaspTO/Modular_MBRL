from planning_module.abstract_best_first_search import AbstractBestFirstSearch
from policy_module.simple_policy import SimplePolicy
from typing import List, Tuple, Dict
from game import Game
from node_module.best_first_node import BestFirstNode
import numpy as np
import random
import warnings






class EpsilonGreedyVisits(SimplePolicy):
    def __init__(self,environment,planning_algorithm,epsilon,reduction='successors'):
        super().__init__(environment,planning_algorithm,reduction=reduction)
        self.epsilon = epsilon

    def play_game(self):
        ''' simple inheritance '''
        return super().play_game()

    def play_move(self,observation,player,mask:np.ndarray):
        ''' returns node of observation and best action of it'''
        if not isinstance(mask,np.ndarray):
            raise ValueError("Mask should be a numpy array")
        node = self.get_planning_algorithm().plan(observation,player,mask)
        if not isinstance(node,BestFirstNode):
            raise ValueError("Epsilon Greedy based on visits needs a best first planning algorithm")
        legal_actions, = np.where(mask == 1)
        if random.random() > self.get_epsilon():
            best_action = max(legal_actions,key=lambda a:node.get_child(a).get_visits())
        else: 
            best_action = random.choice(legal_actions)

        return node,best_action

    def get_epsilon(self):
        return self.epsilon

    def set_epsilon(self,epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return str(self.epsilon) + "-epsilonVisitGreedy"