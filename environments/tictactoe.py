import numpy as np
from copy import deepcopy
from environments.environment import Environment


""" This agent, when playing against expert has 3 difficulties:
0 - only random plays
1 - defends when obvious and attacks randomly
2 - defends and attacks when obvious, random otherwise
"""

class TicTacToe(Environment):
    def __init__(self,self_play=True,expert_start=False,expert_difficulty=2):
        self.self_play = self_play
        self.expert_start = expert_start
        self.board = None
        assert expert_difficulty in [0,1,2]
        self.expert_difficulty = expert_difficulty

    def step(self, action):
        assert not self.done, "can not execute steps when game has finished"
        if self.board is None: raise ValueError("Call Reset first")
        board, reward, done, _ = self._step(action)
        if not self.self_play and not done:
            action = self._expert_action()
            board, reward, done, _ = self._step(action)
            reward = -1 * reward
        return board, reward, done, {} 
        
    def reset(self):
        self.done = False
        self.board = np.zeros((2, 3, 3), dtype="int32")
        self.current_player = 0
        if not self.self_play:
            if self.expert_start:
                action = self._expert_action()
                self._step(action)
            self.expert_start = (self.expert_start == False) #alternate
        return deepcopy(self.board)

    def render(self):
        if self.board is None: raise ValueError("Call Reset first")
        print(self.board[0] - self.board[1])

    def get_action_size(self):
        return 9

    def get_input_shape(self):
        return (2,3,3)

    def get_num_of_players(self):
        if self.self_play:
            return 2
        else:
            return 1

    def get_legal_actions(self):
        if self.board is None: raise ValueError("Call Reset first")
        if self.done: return []
        legal = []
        for action in range(9):
            row, col = self._action_to_pos(action)
            if self.board[0][row, col] == 0 and self.board[1][row, col] == 0:
                legal.append(action)
        return deepcopy(legal)

    def get_current_player(self) -> int:
        if self.board is None: raise ValueError("Call Reset first")
        if self.self_play is False:
            return 0
        else:
            assert self.current_player in [0,1]
            return self.current_player
    

    def _step(self,action):
        if self.done == True:
            raise ValueError("Game is over")
        row,col = self._action_to_pos(action)
        if self.board[0][row,col] != 0 or self.board[1][row,col] != 0:
            raise ValueError("Playing in already filled position")

        self.board[0,row, col] = 1
        self.board = np.array([self.board[1],self.board[0]]) #switch
        self.done = self._have_winner() or len(self.get_legal_actions()) == 0
        reward = 1 if self._have_winner() else 0
        self.current_player = (self.current_player + 1) % 2

        return deepcopy(self.board), reward, self.done, {}

    def _action_to_pos(self,action):
        assert action >= 0 and action <= 8
        row = action // 3
        col = action % 3
        return (row,col)

    def _pos_to_action(self,row,col):
        action = row * 3 + col
        return action

    def _have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[0,i] == 1).all() or (self.board[1,i] == 1).all():
                return True #horizontal
            if (self.board[0,:,i] == 1).all() or (self.board[1,:,i] == 1).all():
                return True #verticals

        #diagonals
        if (self.board[0,0,0] == 1 and self.board[0,1,1] == 1 and self.board[0,2,2] == 1 or \
            self.board[1,0,0] == 1 and self.board[1,1,1] == 1 and self.board[1,2,2] == 1
        ):
            return True 


        if (self.board[0,0,2] == 1 and self.board[0,1,1] == 1 and self.board[0,2,0] == 1 or \
            self.board[1,0,2] == 1 and self.board[1,1,1] == 1 and self.board[1,2,0] == 1
        ):
            return True 

        return False

    def _expert_action(self):
        board = self.board
        summed_board = 1*board[0] + -1*board[1]
        action = np.random.choice(self.get_legal_actions())


        # Horizontal and vertical checks
        if self.expert_difficulty == 2:
            for i in range(3):
                if sum(summed_board[i,:]) == 2: #attacking row position
                    col = np.where(summed_board[i, :] == 0)[0][0]
                    action = self._pos_to_action(i,col)
                    return action

                if sum(summed_board[:,i]) == 2: #attacking col position
                    row = np.where(summed_board[:,i] == 0)[0][0]
                    action = self._pos_to_action(row,i)
                    return action

        if self.expert_difficulty >= 1:
            for i in range(3):
                if sum(summed_board[i,:]) == -2: #defending row position
                    col = np.where(summed_board[i, :] == 0)[0][0]
                    action = self._pos_to_action(i,col)
                    return action

                if sum(summed_board[:,i]) == -2: #defending col position
                    row = np.where(summed_board[:,i] == 0)[0][0]
                    action = self._pos_to_action(row,i)
                    return action

        # Diagonal checks
        diag = summed_board.diagonal()  #left_up-right_dow
        anti_diag = np.fliplr(summed_board).diagonal() #left_down-right_up
        if self.expert_difficulty == 2:
            if sum(diag) == 2:  #attacking diag
                ind = np.where(diag == 0)[0][0]
                row = ind
                col = ind
                action = self._pos_to_action(row,col)
                return action

            if sum(anti_diag) == 2: #attacking anti-diag
                ind = np.where(anti_diag == 0)[0][0]
                row = ind
                col = 2-ind
                action = self._pos_to_action(row,col)
                return action
        
        if self.expert_difficulty >= 1:
            if sum(diag) == -2: #defending diag 
                ind = np.where(diag == 0)[0][0]
                row = ind
                col = ind
                action = self._pos_to_action(row,col)
                return action
                
            if sum(anti_diag) == -2: #defending anti-diag
                ind = np.where(anti_diag == 0)[0][0]
                row = ind
                col = 2-ind
                action = self._pos_to_action(row,col)
                return action

        return action
        
    def __str__(self):
        return "TicTacToe"

