from model_module.query_operations.input_operations.state_op import StateOp
import torch

class MaskOp(StateOp):
    KEY = "MaskOp"
    """ Use an activation function with this - Important to keep this
    consistent so that loss functions now how to update the neural networks """ 
    
    def prediction_query(self,states:torch.tensor,*keys):
        return NotImplementedError
