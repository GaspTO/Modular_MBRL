from planning_module.abstract_depth_first_search import AbstractDepthFirstSearch
from model_module.query_operations.reward_op import RewardOp
from model_module.query_operations.next_state_op import NextStateOp
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
from model_module.query_operations.mask_op import MaskOp
from node_module.node import Node
import numpy as np
import torch



class Minimax(AbstractDepthFirstSearch):
    def __init__(self,
    model,
    action_size,
    num_of_players,
    max_depth,
    invalid_penalty,
    device = None):
        super().__init__(model)
        self.max_depth = max_depth
        self.action_size = action_size
        self.num_of_players = num_of_players
        self.model = model
        self.invalid_penalty = invalid_penalty
        if device is None:
            self.device = torch.device("cpu") #we do not do operations in this algorithm, so just keep it on cpu
        else:
            self.device = device
        print("Minimax is using "+str(self.device)+ " device")
     
    def plan(self,observation,player,mask):
        with torch.no_grad():
            encoded_state, = self.model.representation_query(torch.tensor(np.expand_dims(observation, axis=0)),RepresentationOp.KEY)
            encoded_state = encoded_state.to(self.device)
        node = Node()
        node.set_player(player).set_encoded_state(encoded_state[0]).set_action_mask(mask)
        self._expand_minimax_tree(node) 
        return node

    '''
    Search
    '''
    def _expand_minimax_tree(self,node):
        layers = [[node]]
        for current_depth in range(self.max_depth -1):
            current_layer = []
            self._expand_nodes(layers[-1],estimate=False)
            for n in layers[-1]:
                current_layer.extend(n.get_children_nodes())
            layers.append(current_layer)

        self._expand_nodes(layers[-1],estimate=True)

        for layer in reversed(layers):
            for n in layer:
                self._update_node(n)
        
    def _update_node(self,node):
        best_node,action,value = self._get_best_node(node)
        node.set_value(value)
        
    '''
    Gets best successor of observation 
    ''' 
    
    def _get_best_node(self,node): 
        best_masked_value,best_value, best_action, best_child = float("-inf"),None,None,None
        mask = node.get_action_mask()
        for action,child in node.get_children().items():
            valid = mask[action]
            raw_child_value_to_parent = node.successor_value(action)
            masked_child_value_to_parent = self._value_after_mask(raw_child_value_to_parent,valid)
            if masked_child_value_to_parent > best_masked_value:
                best_masked_value = masked_child_value_to_parent
                best_value, best_action, best_child = raw_child_value_to_parent,action,child
        return best_child,best_action,best_value
    
    def _value_after_mask(self,value,valid):
        return (value * valid) + (1-valid) * self.invalid_penalty
    

    '''
    Expansion and networks
    '''
    def _expand_nodes(self,nodes:list,estimate=False):
        encoded_states = torch.cat([n.get_encoded_state().unsqueeze(0) for n in nodes])
        actions = [list(range(self.action_size))]* len(nodes)
        with torch.no_grad():
            rewards, next_encoded_states = self.model.dynamic_query(encoded_states,actions,RewardOp.KEY,NextStateOp.KEY)
            rewards = rewards.to(self.device)
            next_encoded_states = next_encoded_states.to(self.device)

        if estimate:
            with torch.no_grad():
                values, = self.model.prediction_query(next_encoded_states,StateValueOp.KEY)
                values = values.to(self.device)
                mask = torch.zeros(next_encoded_states.shape[0],self.action_size).to(self.device)  #shouldn't matter
        else:
            values = torch.ones(rewards.shape).to(self.device) * float("Nan")
            with torch.no_grad():
                mask, = self.model.prediction_query(next_encoded_states,MaskOp.KEY) 
                mask = mask.to(self.device)

        flat_idx = 0
        for st_idx in range(len(encoded_states)):
            node = nodes[st_idx]
            for a_idx in range(len(actions[st_idx])):
                action = actions[st_idx][a_idx]
                child_node = Node()
                child_node.set_player((node.get_player() + 1)%self.num_of_players)
                child_node.set_parent_info(node,action,rewards[flat_idx].item())
                child_node.set_value(values[flat_idx].item()).set_encoded_state(next_encoded_states[flat_idx])
                child_node.set_action_mask(mask[flat_idx])
                node.add_child(action,child_node)
                assert (st_idx * self.action_size + a_idx) == flat_idx
                flat_idx += 1
                                        
        
    '''
    Transverses tree, returns a summary string and asserts certain facts for consistency
    '''
    #TODO: Fix Me
    def _transverse(self,node):
        def transversal(node,action,valid,current_depth):
            string = ""
            best_child,best_action,best_value =  self._get_best_node(node)
            depth_string = "(" + str(current_depth) + ") " 
            transition_string =  "|action: " + str(action) + " |valid_bit: " + str(round(valid,2))
            if node.get_parent() is not None:
                parent_string = "|parent_reward:"+ str(round(node.get_parent_reward(),4)) +\
                    " |masked_parent_value:" +str(round(node.get_parent().successor_value(node.get_parent_action())*valid,4)) +\
                    " |raw_parent_value:" +str(round(node.get_parent().successor_value(node.get_parent_action()),4))    
                #assert node.get_parent().successor_value(node.get_parent_action()) <= node.get_parent().get_value()
            else:
                parent_string = ""
            value_string =  "|value: " +str(round(node.get_value(),4)) + " ba:"+str(node.best_action)

            string += "\t" * current_depth + depth_string + " " + transition_string + " " + parent_string + " " + value_string + "\n"

            
            mask = node.get_action_mask()
            for action,child in node.get_children().items():
                string += transversal(child,action,mask[action].item(),current_depth+1)
            return string
        
        string = transversal(node,None,1,0)
        return string


