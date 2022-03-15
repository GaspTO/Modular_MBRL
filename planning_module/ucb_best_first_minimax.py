from planning_module.abstract_best_first_search import AbstractBestFirstSearch
from node_module.best_first_node import BestFirstNode
from model_module.query_operations.reward_op import RewardOp
from model_module.query_operations.next_state_op import NextStateOp
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
from model_module.query_operations.mask_op import MaskOp
from math import sqrt, log
import numpy as np
import torch 



    
class UCBBestFirstMinimax(AbstractBestFirstSearch):
    def __init__(self,
    model,
    action_size,
    num_of_players,
    num_iterations,
    search_expl,
    invalid_penalty,
    device = None):
        super().__init__(model,device)
        self.action_size = action_size
        self.num_of_players = num_of_players
        self.num_iterations = num_iterations
        self.search_expl = search_expl
        self.invalid_penalty = invalid_penalty
        

    def plan(self,observation,player,mask):
        self.model.eval()
        with torch.no_grad():
            encoded_state, = self.model.representation_query(torch.tensor(np.expand_dims(observation, axis=0)),RepresentationOp.KEY)
            encoded_state = encoded_state.to(self.device)
        node = BestFirstNode()
        node.set_player(player).set_encoded_state(encoded_state[0]).set_action_mask(mask)
        for i in range(self.num_iterations):
            self._search_iteration(node)
        return node

    '''
        Main specific algorithm methods
    '''
    def _search_iteration(self,node):
        if node.is_leaf(): 
            self._expand_node(node)
        else:
            best_node = self._get_ucb_best_node(node,exploration=self.search_expl)
            self._search_iteration(best_node)
        self._update_node(node)
       
    def _update_node(self,node):
        best_node = self._get_ucb_best_node(node,exploration=0)
        action = best_node.get_parent_action()
        unmasked_value = node.action_value(action)
        node.set_value(unmasked_value)
        node.increment_visits()

    '''
        Expansion and networks
    '''
    def _expand_node(self,node):
        assert node.get_num_of_children() == 0
        actions = list(range(self.action_size))
        with torch.no_grad():
            rewards, next_encoded_states = self.model.dynamic_query(node.get_encoded_state().unsqueeze(0),[actions],RewardOp.KEY,NextStateOp.KEY)
            rewards = rewards.to(self.device)
            next_encoded_states = next_encoded_states.to(self.device)
            if node.get_action_mask() is None:
                mask, = self.model.prediction_query(node.get_encoded_state().unsqueeze(0),MaskOp.KEY)
                mask = mask.to(self.device)
                node.set_action_mask(mask[0]) 
            else:
                assert node.get_parent() is None
            values, = self.model.prediction_query(next_encoded_states,StateValueOp.KEY)
            values = values.to(self.device)

        for idx in range(len(actions)):
            action = actions[idx]
            child_node = BestFirstNode()                     
            child_node.set_player((node.get_player() + 1)%self.num_of_players)
            child_node.set_parent_info(node,action,rewards[idx].item())
            child_node.set_value(values[idx].item()).set_encoded_state(next_encoded_states[idx])
            node.add_child(action,child_node)
        assert child_node.get_depth() == node.get_depth() + 1
        assert node.get_num_of_children() == self.action_size #!

    
    def _transverse(self,node):
        pr = 0 if node.get_parent_reward() is None else node.get_parent_reward()
        print(node.get_depth()*"\t"+ "("+str(node.get_parent_action())+")" + " value:" + str(round(node.get_value(),2)) + " r:"+str(round(pr,2)) + " t:"+str(round(node.get_value() + pr,2)))
        for n in node.get_children_nodes():
            self._transverse(n)




        



