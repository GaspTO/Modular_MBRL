from model_module.query_operations.reward_op import RewardOp
from model_module.query_operations.next_state_op import NextStateOp
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
from model_module.query_operations.mask_op import MaskOp
from loss_module.loss import Loss
import torch
import numpy as np
from typing import Union, Optional, List, Tuple
import time


class AbstractUnrollingMVR(Loss):
    def __init__(self,
    model,
    unroll_steps,
    gamma_discount=1,
    loss_fun_value=torch.nn.functional.mse_loss, 
    loss_fun_reward=torch.nn.functional.mse_loss,
    loss_fun_mask=torch.nn.functional.binary_cross_entropy,
    coef_loss_value = 1,
    coef_loss_reward = 1,
    coef_loss_mask = 1,
    encoded_state_fidelity = False, #SHOULD OUR STATES MIMICK THE REAL OBSERVATIONS?
    coef_loss_state = 1,
    loss_fun_state=torch.nn.functional.mse_loss,
    average_loss=True,
    device = None):
        self.model = model
        ''' check if model has all the necessary operations '''
        assert isinstance(self.model,RewardOp) and \
            isinstance(self.model,NextStateOp) and \
            isinstance(self.model,StateValueOp) and \
            isinstance(self.model,MaskOp) and \
            isinstance(self.model,RepresentationOp)
        self.unroll_steps = unroll_steps
        self.gamma_discount = gamma_discount
        self.loss_fun_value = loss_fun_value
        self.loss_fun_reward = loss_fun_reward
        self.loss_fun_mask = loss_fun_mask
        self.coef_loss_value = coef_loss_value
        self.coef_loss_reward = coef_loss_reward
        self.coef_loss_mask = coef_loss_mask
        self.encoded_state_fidelity = encoded_state_fidelity
        self.coef_loss_state = coef_loss_state
        self.loss_fun_state = loss_fun_state
        self.average_loss = average_loss
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Loss class is using device: "+str(self.device))

    def get_loss(self,nodes:list,info={}):
        self.model.train()
        if not isinstance(nodes,list): nodes = [nodes]
        loss_reward, loss_value, loss_mask, loss_state, info = self._get_losses(nodes,info)
        loss = loss_value + loss_reward + loss_mask + loss_state 
        assert self.encoded_state_fidelity is False or loss_state == 0
        if self.average_loss:
            loss = loss/len(nodes)
        return loss, info

    def _get_losses(self,nodes:list,info={}):
        """ returns 4 losses: reward, value, masks and state. The state loss is optional, only
        if we want the hidden state of our agent to mimick the real observations. if we want to update
        the states like this, we have to be careful not to make updates to the state when doing backwards 
        on the values, masks and rewards; but only when we do backwards on the state loss """
        if self.encoded_state_fidelity:
            predicted_rewards, predicted_states, actions = self._get_predicted_model(nodes,False)
            predicted_values, predicted_masks = self._unrolled_prediction(predicted_states.detach())
        else:
            predicted_rewards, predicted_states, actions = self._get_predicted_model(nodes,True)
            predicted_values, predicted_masks = self._unrolled_prediction(predicted_states)
        
        #targets
        target_values = self._get_target_values(nodes).float()
        target_rewards = self._get_target_rewards(nodes).float()
        target_masks = self._get_target_masks(nodes).float()
        target_states = self._get_target_states(nodes).float()
        
        #losses
        loss_reward_per_node = torch.stack([self.loss_fun_reward(predicted_rewards[i],target_rewards[i]) for i in range(len(nodes))])
        loss_value_per_node = torch.stack([self.loss_fun_value(predicted_values[i],target_values[i]) for i in range(len(nodes))])
        loss_mask_per_node = torch.stack([self.loss_fun_mask(predicted_masks[i],target_masks[i]) for i in range(len(nodes))])
        
        if self.encoded_state_fidelity:
            loss_state_per_node = torch.stack([self.loss_fun_state(predicted_states[i],target_states[i]) for i in range(len(nodes))]) #!
        else:
            loss_state_per_node = torch.zeros((len(nodes))).to(self.device)
            assert loss_state_per_node.shape == loss_mask_per_node.shape

        total_loss_per_node = loss_mask_per_node + loss_reward_per_node + loss_value_per_node + loss_state_per_node
        loss_reward = torch.sum(loss_reward_per_node) * self.coef_loss_reward 
        loss_value = torch.sum(loss_value_per_node) * self.coef_loss_value
        loss_mask = torch.sum(loss_mask_per_node) * self.coef_loss_mask
        loss_state = torch.sum(loss_state_per_node) * self.coef_loss_state

        #debug info
        info = {"loss_reward_per_node":loss_reward_per_node.detach(), 
                "loss_value_per_node":loss_value_per_node.detach(),
                "loss_mask_per_node": loss_mask_per_node.detach(),
                "loss_per_node":total_loss_per_node.detach(),
                "loss_reward":loss_reward.detach(),
                "loss_value":loss_value.detach(),
                "loss_mask":loss_mask.detach(),
                "loss_state":loss_state.detach(), 
                "predicted_values":predicted_values.detach(),
                "target_values":target_values,
                "predicted_rewards":predicted_rewards.detach(),
                "target_rewards":target_rewards,
                "predicted_masks":predicted_masks.detach(),
                "target_masks":target_masks,
                "predicted_states":predicted_states.detach(),
                "target_states":target_states.detach(), 
                "actions":actions}
        
        return  loss_reward, loss_value, loss_mask , loss_state, info

    def _get_predicted_model(self,nodes:list,state_grad_for_reward):
        predicted_rewards, predicted_states, actions = self._state_reward_unrolling(nodes,self.model,self.unroll_steps,state_grad_for_reward=state_grad_for_reward)
        assert predicted_rewards.shape[1] == self.unroll_steps and predicted_states.shape[1] == self.unroll_steps + 1
        assert predicted_states.shape[0] == predicted_rewards.shape[0] and predicted_rewards.shape[0] == len(nodes)
        return predicted_rewards, predicted_states, actions

    def _state_reward_unrolling(self,nodes:list,model,unroll_steps:int,state_grad_for_reward):
        """ Returns 3 tensors: 
        predicted values -> shape (len(nodes),unroll_step + 1,1)
        predicted_rewards -> shape (len(nodes),unroll_step,1)
        predicted_states -> shape (len(nodes),unroll_step + 1, hidden_state.shape)
        predicted_mask-> shape (len(nodes),unroll_step, action_size) """
        
        total_actions = []
        games = [node.get_game() for node in nodes]
        game_indexes = [node.get_idx_at_game() for node in nodes]
        observations = torch.tensor([games[n_idx].observations[game_indexes[n_idx]] for n_idx in range(len(nodes))],device=self.device)

        current_states, = model.representation_query(observations,RepresentationOp.KEY)
        current_states = current_states.to(self.device)
        predicted_states_list = [current_states]
        predicted_rewards_list = []
        for delta_idx in range(unroll_steps):
            actions = []
            for idx,game in zip(game_indexes,games): #collect actions per game
                if idx + delta_idx < len(game.actions):
                    actions.append([game.actions[idx+delta_idx]])
                else:
                    actions.append([np.random.choice(game.action_size)])

            if not state_grad_for_reward:
                current_states = current_states.detach() 
            predicted_rewards, next_encoded_states = model.dynamic_query(current_states,actions,RewardOp.KEY,NextStateOp.KEY)
            predicted_rewards = predicted_rewards.to(self.device)
            next_encoded_states = next_encoded_states.to(self.device)
            predicted_states_list.append(next_encoded_states)
            predicted_rewards_list.append(predicted_rewards)
            total_actions.append(actions)
            current_states = next_encoded_states
        #swap the two first dimensions, so that the first refers to each node and the second the each unrolling step
        reward_tensor = torch.transpose(torch.stack(predicted_rewards_list),0,1).to(self.device)
        state_tensor = torch.transpose(torch.stack(predicted_states_list),0,1).to(self.device)
        action_tensor = torch.transpose(torch.tensor(total_actions),0,1).to(self.device)
        return reward_tensor, state_tensor, action_tensor

    def _unrolled_prediction(self,encoded_states):
        """ this method takes all the encoded_states in batch and calculates some predictions,
        in this case, it calculates the value and mask for each one of these encoded states. it
        was created to be used after unrolling"""
        model = self.model
        batch = encoded_states.shape[0]
        steps = encoded_states.shape[1]
        ''' first values - we do not want the maks for the first state of each node, only the value'''
        encoded_states1 = encoded_states[:,0:1,]
        flat_encoded_states1 = torch.flatten(encoded_states1,0,1)
        predicted_values_first, = model.prediction_query(flat_encoded_states1,StateValueOp.KEY)
        predicted_values_first = predicted_values_first.to(self.device)
        predicted_values_first = predicted_values_first.view(batch,1,1) #! change last to -1
        ''' second values and masks'''
        encoded_states2 = encoded_states[:,1:] 
        flat_encoded_states2 = torch.flatten(encoded_states2,0,1)
        predicted_values_second, predicted_masks = model.prediction_query(flat_encoded_states2,StateValueOp.KEY,MaskOp.KEY)
        predicted_values_second = predicted_values_second.to(self.device)
        predicted_masks = predicted_masks.to(self.device)
        predicted_values_second = predicted_values_second.view(batch,steps-1,1)
        predicted_masks = predicted_masks.view(batch,steps-1,-1)
        predicted_values = torch.cat((predicted_values_first,predicted_values_second),dim=1) #! change last to -1
        return predicted_values, predicted_masks

    def _get_target_rewards(self,nodes:list):
        """ returns a tensor with shape (len(nodes),unroll_step,1) """
        target_rewards_list = []
        for i,node in enumerate(nodes):
            game = node.get_game()
            idx = node.get_idx_at_game()
            target_rewards = game.rewards[idx:idx+self.unroll_steps] + [0] * (self.unroll_steps-len(game.rewards[idx:idx+self.unroll_steps])) #fill rest with 0s
            target_rewards_list.append(target_rewards)
        return torch.tensor(target_rewards_list,device=self.device).unsqueeze(2)

    ''' masks '''
    def _get_target_masks(self,nodes:list):
        """ learns the mask for the next unroll_steps states after the node. 
        It does not learn the mask for the current observation, since that is given by the environment,
        but only the next unrolled states. 
        returns shape (len(nodes),unroll_step, action_size) """
        total_masks = []
        for node in nodes:
            game = node.get_game()
            idx = node.get_idx_at_game()
            masks = []
            for delta_idx in range(1,self.unroll_steps+1): #starts at one because we do not need to learn the mask for the real observation
                if (idx + delta_idx) < len(game.nodes):
                    mask = game.nodes[idx + delta_idx].get_action_mask().to(self.device)
                else:
                    mask = torch.tensor([0]*game.action_size,device=self.device) #redundant
                masks.append(mask)
            total_masks.append(torch.stack(masks))
        return torch.stack(total_masks)

    def _get_target_states(self,nodes:list):
        """ returns shape (len(nodes),unroll_step + 1, hidden_state.shape) """
        total_observations = []
        for node in nodes:
            game = node.get_game()
            idx = node.get_idx_at_game()
            observations = []
            for delta_idx in range(0,self.unroll_steps+1): #starts at one because we do not need to learn the mask for the real observation
                if (idx + delta_idx) < len(game.nodes):
                    obs = torch.tensor(game.observations[idx + delta_idx],device=self.device)
                else:
                    obs = torch.tensor(game.observations[-1],device=self.device) #! JUST THE LAST ONE WITH VALUE 0
                observations.append(obs)
            total_observations.append(torch.stack(observations))
        return torch.stack(total_observations)


    def _get_target_values(self,nodes:list):
        """  Extend this class to calculate the value in the way you want it"""
        raise NotImplementedError
