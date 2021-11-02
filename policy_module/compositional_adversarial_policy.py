from game import Game
from node_module.node import Node 
from policy_module.simple_policy import SimplePolicy
from typing import List, Tuple, Dict
from copy import deepcopy


''' Beta version '''
class CompositionalAdversarialPolicy(SimplePolicy):
    def __init__(self,environment,policy1,policy2,reduction='successors',debug=False):
        super().__init__(environment,None,reduction,debug)
        self.policy1 = policy1
        self.policy2 = policy2
        self.player0 = self.policy1
        self.player1 = self.policy2

    def play_game(self) -> Game:
        ret = super().play_game()
        self.player0, self.player1 = self.player1, self.player0
        return ret

    def play_move(self,observation,player,mask) -> Tuple[Node,int]:
        if player == 0:
            ret = self.player0.play_move(observation,player,mask)
        elif player == 1:
            ret = self.player1.play_move(observation,player,mask)
        return ret

    ''' class specific '''
    def  get_players(self):
        return [self.player0,self.player1]
       
        

