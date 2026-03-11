from rl_base import RLBase
import numpy as np
import random
from collections import defaultdict


class DoubleQLearning(RLBase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.q1 = defaultdict(lambda: np.zeros(self.num_actions))
        self.q2 = defaultdict(lambda: np.zeros(self.num_actions))

    def get_discretize_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        q_total = self.q1[state] + self.q2[state]

        return np.argmax(q_total)

    def update(self, state, action, reward, next_state):

        if random.random() < 0.5:

            best = np.argmax(self.q1[next_state])

            target = reward + self.gamma * self.q2[next_state][best]

            self.q1[state][action] += self.lr * (
                target - self.q1[state][action]
            )

        else:

            best = np.argmax(self.q2[next_state])

            target = reward + self.gamma * self.q1[next_state][best]

            self.q2[state][action] += self.lr * (
                target - self.q2[state][action]
            )