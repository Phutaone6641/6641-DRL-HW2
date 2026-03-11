import numpy as np
import pickle
from collections import defaultdict


class RLBase:

    def __init__(
        self,
        control_type,
        num_actions,
        action_range,
        discretize_state_weight,
        learning_rate,
        initial_epsilon,
        epsilon_decay_rate,
        final_epsilon,
        discount_factor
    ):

        # algorithm type
        self.control_type = control_type

        # action parameters
        self.num_actions = num_actions
        self.action_low = action_range[0]
        self.action_high = action_range[1]

        # discretization
        self.state_weight = discretize_state_weight

        # learning parameters
        self.lr = learning_rate
        self.gamma = discount_factor

        # epsilon-greedy
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay_rate
        self.final_epsilon = final_epsilon

        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    # -------------------------------------------------
    # ε-greedy action selection
    # -------------------------------------------------

    def get_discretize_action(self, state):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        return np.argmax(self.q_table[state])

    # -------------------------------------------------
    # convert discrete action → continuous action
    # -------------------------------------------------

    def mapping_action(self, action_index):

        step = (self.action_high - self.action_low) / (self.num_actions - 1)

        return self.action_low + action_index * step

    # -------------------------------------------------
    # discretize continuous state
    # -------------------------------------------------

    def discretize_state(self, observation):

        state = tuple(
            np.round(np.array(observation) * self.state_weight).astype(int)
        )

        return state

    # -------------------------------------------------
    # epsilon decay
    # -------------------------------------------------

    def decay_epsilon(self):

        self.epsilon = max(
            self.final_epsilon,
            self.epsilon * self.epsilon_decay
        )

        return self.epsilon

    # -------------------------------------------------
    # save Q-table
    # -------------------------------------------------

    def save_q_value(self, path):

        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    # -------------------------------------------------
    # load Q-table
    # -------------------------------------------------

    def load_q_value(self, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(
            lambda: np.zeros(self.num_actions),
            data
        )