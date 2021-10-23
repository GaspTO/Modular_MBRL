from typing import List, Tuple, Dict
import numpy as np
from abc import abstractmethod

"""
Environments work with numpy arrays, so don't forget to convert them to torch tensors when appropriate
"""
class Environment:

    def step(self,action:int) -> Tuple[np.ndarray,float,bool,Dict]:
        """ return next_observation, reward, done, info.""" 
        raise NotImplementedError

    def reset(self) -> Tuple[np.ndarray,int,np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def render(self) -> None:
        raise NotImplementedError

    def get_action_size(self) -> int:
        raise NotImplementedError

    def get_input_shape(self) -> Tuple[int]:
        raise NotImplementedError

    def get_num_of_players(self) -> int:
        """ default for single player environments """
        return 1  

    def get_legal_actions(self) -> List[int]:
        "return an empty list when environment has reached the end, for consistency"
        raise NotImplementedError

    def get_action_mask(self) -> np.ndarray:
        legal_actions = self.get_legal_actions()
        mask = np.zeros(self.get_action_size())
        mask[legal_actions] = 1
        assert (np.where(mask == 1)[0] == legal_actions).all()
        return mask       

    def get_current_player(self) -> int:
        """ default for single player environments.
        return a player even when environment has reached the end, for consistency """
        return 0 



    