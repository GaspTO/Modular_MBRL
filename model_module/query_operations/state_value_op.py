from model_module.query_operations.input_operations.state_op import StateOp
import torch

class StateValueOp(StateOp):
    KEY = "StateValueOp"
    
    def prediction_query(self,states:torch.tensor,*keys):
        return NotImplementedError
