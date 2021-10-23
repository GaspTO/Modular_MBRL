import torch
from typing import List
from model_module.operation import Operation

class StateActionOp(Operation):

	def dynamic_query(self,states:torch.tensor,actions:List[list],*keys):
		""" The output should have only two major dimensions. For instance,
    	a states input with shape (2,...) and actions [[0,1,2,3,4],[1,2]], should return an output
    	with shape (7,...). """
		return NotImplementedError
