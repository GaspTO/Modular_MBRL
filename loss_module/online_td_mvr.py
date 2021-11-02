from loss_module.abstract_unrolling_mvr import AbstractUnrollingMVR
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
import torch
import numpy as np


'''
    unroll_steps = number of states to predict
    n_steps = the amount of steps to bootstrap in each unrolled step
'''
class OnlineTDMVR(AbstractUnrollingMVR):
    def __init__(
    self,
    model,
    unroll_steps,
    n_steps,
    gamma_discount=1,
    loss_fun_value=torch.nn.functional.mse_loss,
    loss_fun_reward=torch.nn.functional.mse_loss,
    loss_fun_mask=torch.nn.functional.binary_cross_entropy,
    coef_loss_value=1,
    coef_loss_reward=1,
    coef_loss_mask=1,
    encoded_state_fidelity = False, #SHOULD OUR STATES MIMICK THE REAL OBSERVATIONS?
    coef_loss_state = 1,
    loss_fun_state=torch.nn.functional.mse_loss,
    average_loss=True,
    device = None):
        super().__init__(
            model=model,
            unroll_steps=unroll_steps,
            gamma_discount=gamma_discount,
            loss_fun_value=loss_fun_value,
            loss_fun_reward=loss_fun_reward,
            loss_fun_mask=loss_fun_mask,
            coef_loss_value=coef_loss_value,
            coef_loss_reward=coef_loss_reward,
            coef_loss_mask=coef_loss_mask,
            encoded_state_fidelity=encoded_state_fidelity,
            coef_loss_state=coef_loss_state,
            loss_fun_state=loss_fun_state,
            average_loss=average_loss,
            device = None)
        self.n_steps = n_steps


    def _get_target_values(self,nodes:list):
        return self._get_bootstrapped_target_values(nodes)
    
    def _get_bootstrapped_target_values(self,nodes:list):
        observations_to_estimate = self._collect_bootstrapped_observations(nodes)

        ''' Estimate the bootstrapping observations collected '''
        raw_values = []
        if len(observations_to_estimate) > 0:
            with torch.no_grad():
                states, = self.model.representation_query(observations_to_estimate,RepresentationOp.KEY)
                raw_values, = self.model.prediction_query(states,StateValueOp.KEY)
                raw_values = raw_values.to(self.device)

        ''' Create target values based on rewards and bootstrapping steps '''
        target_values = []
        raw_values_idx = 0  
        for node in nodes:
            target_values_per_node = []
            game = node.get_game()
            idx = node.get_idx_at_game()
            for current_index in range(0,self.unroll_steps+1):
                bootstrap_index = current_index + self.n_steps
                
                if bootstrap_index < (len(game.nodes)-idx): #bootstrapped value
                    raw_value = raw_values[raw_values_idx]
                    assert (torch.from_numpy(game.observations[idx+bootstrap_index])  == observations_to_estimate[raw_values_idx]).all()
                    raw_values_idx += 1
                    value = raw_value * self.gamma_discount**self.n_steps
                    if game.players[idx+current_index] != game.players[idx+bootstrap_index]:
                        value = -value
                else:
                    value = 0

                for i, reward in enumerate(game.rewards[idx+current_index:idx+bootstrap_index]): 
                    if game.players[idx+current_index] == game.players[idx+current_index+i]:
                        value += reward * self.gamma_discount**i
                    else:
                        value -= reward * self.gamma_discount**i
                target_values_per_node.append([value])
            target_values.append(target_values_per_node)
        target_values = torch.tensor(target_values,device=self.device)
        assert raw_values_idx == len(raw_values)
        return target_values

    def _collect_bootstrapped_observations(self,nodes:list):
        """ Collect bootstrapping observations into a batch
        for more efficiency in the model """
        observations_to_estimate = []
        for node in nodes:
            game = node.get_game()
            idx = node.get_idx_at_game()
            current_index = 0
            bootstrap_index = current_index + self.n_steps
            while (current_index < self.unroll_steps + 1) and (idx+bootstrap_index < len(game.nodes)):
                obs = game.observations[idx+bootstrap_index]
                observations_to_estimate.append(obs)
                current_index += 1
                bootstrap_index = current_index + self.n_steps
        return torch.tensor(observations_to_estimate,self.device)


    def __str__(self):
        return "Online_TD_Loss" + "_unroll"+str(self.unroll_steps)+"_nsteps"+str(self.n_steps)
