from policy_module.simple_policy import SimplePolicy
from typing import List, Tuple, Dict
from node_module.best_first_node import BestFirstNode
import numpy as np



class VisitRatio(SimplePolicy):
    def __init__(self,environment,planning_algorithm,temperature=1,reduction='successors'):
        super().__init__(environment,planning_algorithm,reduction=reduction)
        self.temperature = temperature

    def play_game(self):
        ''' simple inheritance '''
        return super().play_game()

    def play_move(self, observation,player,mask:np.ndarray):
        ''' returns node of observation and best action of it'''
        if not isinstance(mask,np.ndarray):
            raise ValueError("Mask should be a numpy array")
        node = self.get_planning_algorithm().plan(observation,player,mask)
        if not isinstance(node,BestFirstNode):
            raise ValueError("Exponentiated Visit Count based needs a best first planning algorithm")
        probabilities = []
        legal_actions, = np.where(mask == 1)
        for action in legal_actions:
            probabilities.append(node.get_child(action).get_visits())
        probabilities = np.array(probabilities) ** (1/self.get_temperature())
        probabilities =  probabilities/probabilities.sum()
        action = np.random.choice(legal_actions,p=probabilities)
        return node,action

    ''' instance specific '''
    def get_temperature(self):
        return self.temperature

    def set_temperature(self,temperature):
        self.temperature = temperature
        
    def __str__(self):
        return str(self.get_temperature()) + "-exponentiatedVisitCount"