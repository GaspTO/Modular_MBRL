from planning_module.abstract_best_first_search import AbstractBestFirstSearch
from model_module.query_operations.reward_op import RewardOp
from model_module.query_operations.next_state_op import NextStateOp
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
from model_module.query_operations.mask_op import MaskOp
from node_module.mcts_node import MCTSNode
from math import sqrt, log
import torch




class UCBMonteCarloTreeSearch(AbstractBestFirstSearch):
    def __init__(self,
    model,
    action_size,
    num_of_players,
    num_iterations,
    search_expl,
    invalid_penalty):
        super().__init__(model)
        self.action_size = action_size
        self.num_of_players = num_of_players
        self.num_iterations = num_iterations
        self.search_expl = search_expl
        self.invalid_penalty = invalid_penalty

    def plan(self,observation,player,mask):
        self.player = player
        with torch.no_grad():
             encoded_state, = self.model.representation_query(torch.tensor([observation]),RepresentationOp.KEY)
             value, = self.model.prediction_query(encoded_state,StateValueOp.KEY)
        node = MCTSNode()
        node.set_player(self.player)
        node.add_value(value.item())
        node.set_encoded_state(encoded_state[0]).set_action_mask(mask)
        for i in range(self.num_iterations):
            self._search_iteration(node)
        return node

    '''
        Main specific algorithm methods
    '''
    def _search_iteration(self,node):
        if node.is_leaf(): 
            self._expand_node(node)
            rollout = self._get_children_average_successor_value(node,strict=True)
        else:
            best_node = self._get_ucb_best_node(node,exploration=self.search_expl)
            rollout = self._search_iteration(best_node)
        self._update_node(node,rollout)
        rollout = self._increment_rollout(node,rollout)
        return rollout

    def _increment_rollout(self,node,rollout):
        if node.get_parent() is not None:
            if node.get_parent().get_player() != node.get_player():
                rollout = node.get_parent_reward() - rollout 
            else:
                rollout = node.get_parent_reward() + rollout 
        return rollout

    def _update_node(self,node,rollout):
        node.add_value(rollout)
        node.increment_visits()
        
    '''
        Expansion and networks
    '''
    def _expand_node(self,node):
        assert node.get_num_of_children() == 0

        actions = list(range(self.action_size))
        with torch.no_grad():
            rewards, next_encoded_states = self.model.dynamic_query(node.get_encoded_state().unsqueeze(0),[actions],RewardOp.KEY,NextStateOp.KEY) 
            if node.get_action_mask() is None:
                assert node.get_parent() is not None
                mask, = self.model.prediction_query(node.get_encoded_state().unsqueeze(0),MaskOp.KEY) 
                node.set_action_mask(mask[0])
            else:
                assert node.get_parent() is None
            values, = self.model.prediction_query(next_encoded_states,StateValueOp.KEY)
            
        for idx in range(len(actions)):
            action = actions[idx]
            child_node = MCTSNode()
            child_node.set_player((node.get_player() + 1)%self.num_of_players)
            child_node.set_parent_info(node,action,rewards[idx].item())
            child_node.add_value(values[idx].item())
            child_node.set_encoded_state(next_encoded_states[idx])
            node.add_child(action,child_node)
        assert node.get_num_of_children() == self.action_size
        assert node.get_player() >= 0 and node.get_player() < self.num_of_players


    def _get_children_average_successor_value(self,node,strict=True):
        # don't use penalty here
        total_value = 0.
        total_mask = 0.
        mask = node.get_action_mask()
        for child in node.get_children_nodes():
            total_value += node.successor_value(child.get_parent_action()) * mask[child.get_parent_action()].item() 
            total_mask += mask[child.get_parent_action()].item()
            if strict: assert child.get_visits() == 0
        return total_value/total_mask if total_mask != 0. else 0.

    def _transverse(self,node):
        pr = 0 if node.get_parent_reward() is None else node.get_parent_reward()
        print(node.get_depth()*"\t"+ "("+str(node.get_parent_action())+")" + " value:" + str(round(node.get_value(),2)) + " r:"+str(round(pr,2)) + " t:"+str(round(node.get_value() + pr,2)))
        for n in node.get_children_nodes():
            self._transverse(n)
    

