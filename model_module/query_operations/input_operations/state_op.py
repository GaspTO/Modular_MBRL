import torch
from model_module.operation import Operation

class StateOp(Operation):

    def prediction_query(self,states:torch.tensor,*keys):
        return NotImplementedError
