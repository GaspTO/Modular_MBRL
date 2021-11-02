from loss_module.loss import Loss
from loss_module.abstract_unrolling_mvr import AbstractUnrollingMVR
import torch


class MonteCarloMVR(AbstractUnrollingMVR):
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
    encoded_state_fidelity = False,
    coef_loss_state = 1,
    loss_fun_state=torch.nn.functional.mse_loss,
    average_loss=True,
    device = None):
        super().__init__(model=model,
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
    
    def _get_target_values(self,nodes:list):
        return self.discounted_accumulated_rewards(nodes,self.unroll_steps,self.gamma_discount)

    def discounted_accumulated_rewards(self,nodes:list,unroll_steps,gamma_discount=1)->torch.tensor:
        accumulated_rewards = []    
        for node in nodes:
            game = node.get_game()
            idx = node.get_idx_at_game()
            total_reward = 0
            accumulated_rewards_per_game = []
            assert len(game.rewards) + 1 == len(game.players)
            for i in range(len(game.rewards)-1,idx-1,-1):
                if game.players[i] == game.players[i+1]:
                    total_reward = total_reward * gamma_discount
                else:
                    total_reward = -total_reward * gamma_discount
                total_reward += game.rewards[i]
                accumulated_rewards_per_game.insert(0,[total_reward])

            if len(accumulated_rewards_per_game) >= unroll_steps+1:
                accumulated_rewards_per_game = accumulated_rewards_per_game[:unroll_steps+1]
            else:
                accumulated_rewards_per_game = accumulated_rewards_per_game + [[0.0]]*(unroll_steps+1-len(accumulated_rewards_per_game))
            assert len(accumulated_rewards_per_game) == unroll_steps+1
            accumulated_rewards.append(accumulated_rewards_per_game)
        return torch.tensor(accumulated_rewards,device=self.device)


    def __str__(self):
        return "Monte_Carlo_loss" + "_unroll" + str(self.unroll_steps)


