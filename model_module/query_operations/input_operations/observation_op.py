import torch
from model_module.operation import Operation

class ObservationOp(Operation):
    
    def representation_query(self,observations,*args) -> torch.tensor:
        return NotImplementedError