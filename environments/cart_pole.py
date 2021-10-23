from environments.environment import Environment
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv



'''
Adapted from https://github.com/werner-duvaud/muzero-general
'''

class CartPole(Environment):
    def __init__(self,max_steps=1000):
        self.environment = gym.make("CartPole-v1")
        self.max_steps = max_steps

    def step(self,action):
        assert not self.done, "can not execute steps when game has finished"
        assert self.steps_taken < self.max_steps
        self.steps_taken +=1
        obs, reward, self.done, info = self.environment.step(action)
        if self.steps_taken == self.max_steps:
            self.done = True
        self.current_observation = obs
        return obs, reward, self.done, info
        
    def reset(self):
        self.done = False
        self.steps_taken = 0
        self.current_observation = self.environment.reset()
        return self.current_observation

    def close(self):
        self.environment.close()

    def render(self):
        self.environment.render()

    def get_action_size(self):
        return 2

    def get_input_shape(self):
        return (4,)

    def get_legal_actions(self):
        """ In CartPole, the two actions are always legal """
        if not self.done:
            return [0,1]
        else:
            return []

    def __str__(self):
        return "CartPole-v1"