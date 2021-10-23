
import torch

class SimpleOptimizer:
    def __init__(self,parameters,optimizers:list,schedulers:list=[],max_grad_norm=20):
        if not isinstance(optimizers,list): optimizers = [optimizers]
        if not isinstance(schedulers,list): schedulers = [schedulers]
        self.parameters = parameters
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.max_grad_norm = max_grad_norm
        
    def optimize(self,loss):
        if loss.grad_fn is None:
            return 0.
        for optim in self.optimizers:
            optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm) #clip gradients to help stabilise training
        for optim in self.optimizers:
            optim.step()
        total_norm = 0
        for p in self.parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        for scheduler in self.schedulers:
            scheduler.step()
        return total_norm
