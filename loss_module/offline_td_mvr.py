from loss_module.loss import Loss
from loss_module.monte_carlo_mvr import MonteCarloMVR

import torch
import random

class OfflineTDMVR(MonteCarloMVR):
    def __init__(self,
    model,
    unroll_steps,
    n_steps,
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
            device = device)
        self.n_steps = n_steps

    def _get_target_values(self,nodes:list):
        return self._get_node_bootstrapped_target_values(nodes)

    def _get_node_bootstrapped_target_values(self,nodes:list):
        """ for each of the unroll steps, get sum of rewards for n_steps and then get the n_step node.get_value()
        TODO: this is a bit inneficient. Not too much, but... enough """
        total_target_values = []
        for node in nodes:
            game = node.get_game()
            idx = node.get_idx_at_game()
            target_values = []
            for current_index in range(0,self.unroll_steps+1):
                bootstrap_index = current_index + self.n_steps
                if  idx + bootstrap_index < len(game.nodes): #get bootstrapped value
                    value = game.nodes[idx+bootstrap_index].get_value() * self.gamma_discount**self.n_steps 
                    if game.players[idx+current_index] != game.players[idx+bootstrap_index]:
                        value = -value
                else:
                    value = 0

                #collect reward sum until bootstrapped value 
                for i, reward in enumerate(game.rewards[idx+current_index:idx+bootstrap_index]): 
                    if game.players[idx+current_index] == game.players[idx+current_index+i]:
                        value += reward * self.gamma_discount**i
                    else:
                        value -= reward * self.gamma_discount**i

                if current_index < len(game.observations[idx:]): #is current index imaginary?
                    target_values.append(value)
                else:
                    assert value == 0
                    target_values.append(0)
            total_target_values.append(target_values)
        return torch.tensor(total_target_values,device=self.device).unsqueeze(2)
    

    def __str__(self):
        return "Offline_TD_Loss" + "_unroll"+str(self.unroll_steps)+"_nsteps"+str(self.n_steps)