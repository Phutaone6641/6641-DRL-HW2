from rl_base import RLBase
import numpy as np


class QLearning(RLBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, state, action, reward, next_state):

        best_next = np.max(self.q_table[next_state])

        td_target = reward + self.gamma * best_next

        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.lr * td_error