import torch
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
from torch import nn


class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, input):
        """
        Forward pass of the ANN.
        Defines how input data is processed through the network.
        input -> fc1 -> relu -> fc2 -> output
        """
        x = self.fc1(input)
        x = F.relu(x)
        return self.fc2(x)


class ReplayMemory:
    def __init__(self, capacity):
        """Initialise the replay memory."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        """
        Push an event into the memory.
        """
        # event - tuple of (state, action, reward, next_state, done)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """
        Sample a batch of events from the memory.
        """
        # experiences is a list of tuples of
        # (state, action, reward, next_state, done)
        # 1 example is (state, action, reward, next_state, done)
        experiences = random.sample(self.memory, batch_size)
        states = (
            torch.from_numpy(np.vstack(
                [e[0] for e in experiences if e is not None]
                )).float().to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack(
                [e[1] for e in experiences if e is not None]
                )).float().to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack(
                [e[2] for e in experiences if e is not None]
                )).float().to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack(
                [e[3] for e in experiences if e is not None]
                )).float().to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack(
                [e[4] for e in experiences if e is not None]
                ).astype(np.uint8)
            ).float().to(self.device)
        )
        return states, actions, rewards, next_states, dones


# Hyperparameters
episodes = 10000  # number of episodes to train for
max_steps = 200000  # max number of steps per episode
epsilon_start = 1.0  # initial epsilon for exploration-exploitation trade-off
epsilon_end = 0.001  # final epsilon for exploration-exploitation trade-off
epsilon_decay = 0.9995  # decay rate for epsilon
learning_rate = 0.001  # rate at which the agent updates its weights
minibatch_size = 100  # number of samples used in each training step
gamma = 0.95  # discount factor for future rewards
replay_memory_capacity = int(1e5)  # maximum capacity of the replay memory
interpolation_steps = 1e-2  # steps to interpolate target and online network
input_size = 16  # number of input features
output_size = 4  # number of possible actions
scores_of_episodes = deque(maxlen=100)


class Agent:
    def __init__(self, input_size, output_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.model = ANN(input_size, output_size).to(self.device)
        self.target_model = ANN(input_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
            )
        self.memory = ReplayMemory(replay_memory_capacity)
        self.t_steps = 0
        self.recorded_scores = -1
        self.epsilon = -1

    def get_state(self, game):
        """Get the state of the game."""
        head_x, head_y = game.snake.x[0], game.snake.y[0]

        point_left = [(head_x - game.grid_size), head_y]
        point_right = [(head_x + game.grid_size), head_y]
        point_up = [head_x, (head_y - game.grid_size)]
        point_down = [head_x, (head_y + game.grid_size)]

        # Find the closest green apple
        closest_green = min(
            game.green_apples,
            key=lambda a: abs(a.x - head_x) + abs(a.y - head_y)
        )

        state = [
            # is_danger will be implemented later
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            # move direction
            game.snake.direction == "LEFT",
            game.snake.direction == "RIGHT",
            game.snake.direction == "UP",
            game.snake.direction == "DOWN",
            # closest green apple position relative to head
            closest_green.x < head_x,
            closest_green.x > head_x,
            closest_green.y < head_y,
            closest_green.y > head_y,
            # red apple position relative to head
            game.red_apple.x < head_x,
            game.red_apple.x > head_x,
            game.red_apple.y < head_y,
            game.red_apple.y > head_y,
        ]

        return np.array(state, dtype=int)
