from model_module.query_operations.input_operations.observation_op import ObservationOp
import torch

class RepresentationOp(ObservationOp):
    KEY = "RepresentationOp"

    def representation_query(self, observations, *key) -> torch.tensor:
        return NotImplementedError
