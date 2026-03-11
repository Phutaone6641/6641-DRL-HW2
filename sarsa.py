from rl_base import RLBase


class SARSA(RLBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, state, action, reward, next_state, next_action):

        td_target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.lr * td_error