from environments.environment import Environment
import gym
import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')

class Breakout(Environment):
    def __init__(self):
        self.environment = gym.make("Breakout-v4")
        self.done = True

    def step(self, action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        if self.done is True:
            raise ValueError("The game has ended already, call reset")
        observation, reward, done, info = self.environment.step(action)
        #observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = np.asarray(observation, dtype="float32") / 255.0
        #observation = np.moveaxis(observation, -1, 0)
        assert self.get_input_shape() == observation.shape
        return observation, reward, done, info
        
    def reset(self):
        """
        Reset the game for a new game.
        Returns:
            Initial observation of the game.
        """
        self.done = False
        observation = self.environment.reset()
        #observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = np.asarray(observation, dtype="float32") / 255.0
        #observation = np.moveaxis(observation, -1, 0)
        assert self.get_input_shape() == observation.shape
        return observation

    def close(self):
        self.environment.close()

    def render(self):
        self.environment.render()

    def get_action_size(self):
        return 4

    def get_input_shape(self):
        return (210,160,3)

    def get_legal_actions(self):
        """ In CartPole, the two actions are always legal """
        if not self.done:
            return [0,1]
        else:
            return []

    def get_legal_actions(self):
        return [0,1,2,3]

    def __str__(self):
        return "Breakout-v4"