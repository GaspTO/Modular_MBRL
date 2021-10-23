import gym
from gym.core import Env
try:
    import gym_minigrid
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install gym_minigrid"')
import numpy as np
from environments.environment import Environment
import random


'''
Original gym environment: https://github.com/maximecb/gym-minigrid

set agent_start_pos to None for it to be a random every time you reset
'''


class Minigrid(Environment):
    def __init__(self,N=6,reward_scaling=1, max_steps=None,agent_start_pos=(1,1),seed=None):
        self.reward_scaling = reward_scaling
        self.max_steps = max_steps
        self.environment = gym_minigrid.envs.empty.EmptyEnv(size=N+2,agent_start_pos=agent_start_pos)
        self.environment = gym_minigrid.wrappers.ImgObsWrapper(self.environment)
        if seed is not None:
            self.environment.seed(seed)

    def step(self, action):
        assert not self.done, "can not execute steps when game has finished"
        assert self.steps_taken < self.max_steps
        assert action in [0,1,2]
        self.steps_taken +=1
        obs, reward, self.done, info = self.environment.step(action)
        if self.steps_taken == self.max_steps:
            self.done = True
        if reward > 0:
            #Ths minigrid gives a reward according to how many steps it took before, 
            #which goes against the markov property
            reward = 1
        return obs, self.reward_scaling*reward, self.done, info 

    def reset(self):
        self.done = False
        self.steps_taken = 0
        return np.array(self.environment.reset())

    def close(self):
        self.environment.close()

    def render(self):
        return self.environment.render()

    def get_action_size(self):
        return 3

    def get_input_shape(self):
        return (7,7,3)

    def get_num_of_players(self):
        return 1
    
    def get_legal_actions(self):
        if self.done:
            return []
        else:
            return [0,1,2]    

    def __str__(self):
        return "MiniGrid"

 
