from game import Game
from node_module.node import Node 
from policy_module.policy import Policy
from typing import List, Tuple, Dict
from copy import deepcopy



class SimplePolicy(Policy):
    def __init__(self,environment,planning,reduction='successors',debug=False):
        super().__init__(environment)
        self.planning_algorithm = planning
        self.reduction_operations = ["root","successors","none"]
        assert reduction in self.reduction_operations, "reduction needs to be in" + str(self.reduction_operations)
        self.reduction = reduction
        self.debug = debug
        self.info = {}
        

    def play_game(self) -> Game:
        """ override this method if necessary, but this one should be good enough 
            for most applications. It iterates through the environment, using the planning
            the choose a decision. At each step, it fills the game and node with the appropriate information
        """
        current_observation = self.environment.reset()
        player = self.environment.get_current_player()
        mask = self.environment.get_action_mask()
        game = Game(self.environment.get_input_shape(),self.environment.get_action_size(),self.environment.get_num_of_players())
        game.observations.append(current_observation)
        game.players.append(player)
        game.masks.append(mask)
        done = False
        while not done:
            if self.debug:
                env = deepcopy(self.environment)
                env.render()
                input("Press Ok for next step")
                env.close()
            node,action = self.play_move(current_observation,player,mask)
            current_observation, reward , done , info = self.environment.step(action)
            player = self.environment.get_current_player()
            mask = self.environment.get_action_mask()
           
            self._reduce_node(node)
            #set info
            node.set_game(game,len(game.nodes))
            game.observations.append(current_observation)
            game.players.append(player)
            game.masks.append(mask)
            game.nodes.append(node)
            game.actions.append(action)
            game.rewards.append(reward)
            game.dones.append(done)
            game.infos.append(info)
        return game


    def _reduce_node(self,node):
        """ cut some successors of the node, to save memory """
        if self.reduction == "root": #cut root successors
            node.detach_from_tree()
        elif self.reduction == "successors": #cut successors' successors
            for succ in node.get_children_nodes():
                succ.detach_from_tree()
        elif self.reduction == "none": #don't do anything
            return 
        else:
            raise ValueError("reduction operation is invalid")

    def get_planning_algorithm(self):
        return self.planning_algorithm

    def set_planning_algorithm(self,planning):
        self.planning_algorithm = planning


