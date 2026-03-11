from rl_base import RLBase


class MonteCarlo(RLBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, episode):

        G = 0

        for state, action, reward in reversed(episode):

            G = reward + self.gamma * G

            self.q_table[state][action] += self.lr * (
                G - self.q_table[state][action]
            )