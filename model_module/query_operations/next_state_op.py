from typing import List
from model_module.query_operations.input_operations.state_action_op import StateActionOp
import torch


class NextStateOp(StateActionOp):
    KEY = "StateActionOp"

    def dynamic_query(self,states:torch.tensor,actions:List[list],*keys):
        return NotImplementedError
    
