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
        # determine the device to use for the memory (GPU or CPU)
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
        # local network for Q-values prediction of current state
        self.local_network = ANN(input_size, output_size).to(self.device)
        # target network for stable Q-values computation
        self.traget_network = ANN(input_size, output_size).to(self.device)
        # copy the weights from the local network to the target network
        self.traget_network.load_state_dict(self.local_network.state_dict())
        # Adam optimizer for local network, a gradient descent that:
        # - uses an adaptive learning rate for each parameter
        # - uses a momentum term to accelerate the convergence
        # - uses a decay rate for the learning rate
        # - uses a decay rate for the momentum term
        # - uses a decay rate for the decay rate
        # - uses a decay rate for the decay rate
        self.optimizer = torch.optim.Adam(
            self.local_network.parameters(), lr=learning_rate
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

    def step(self, state, action, reward, next_state, done):
        """Take a step in the environment."""
        self.memory.push((state, action, reward, next_state, done))
        self.t_steps = (self.t_steps + 1) % 4
        if self.t_steps == 0:
            experiences = self.memory.sample(minibatch_size)
            self.learn(experiences)  # will be implemented later

    def get_action(self, state, epsilon):
        """Get the action for the given state."""
        # .unsqueeze(0) — adds a batch dimension,
        # changing shape from [16] to [1, 16]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        # disable gradient calculation to speed up inference
        with torch.no_grad():
            # since action size is 4 we will get:
            # [Q(state, action_0), ..., Q(state, action_3)]
            actions = self.local_network(state)
        # put the local_network back in training mode
        self.local_network.train()
        # epsilon-greedy policy:
        # with probability epsilon, explore randomly
        # with probability 1-epsilon, exploit the best action
        if random.random() > epsilon:
            # get the action with the highest Q-value
            action = torch.argmax(actions).item()
        else:
            # explore randomly
            action = random.randint(0, 3)
        return action

    def learn(self, experiences):
        """Learn from the experiences."""
        states, actions, rewards, next_states, dones = experiences
        next_actions = (
            self.traget_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        # compute the Q-targets using the Bellman equation:
        q_targets = rewards + gamma * next_actions * (1 - dones)
        # compute the Q-values for the current state
        q_values = self.local_network(states).gather(1, actions)
        # compute the loss
        loss = F.mse_loss(q_values, q_targets)
        # optimize the local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the target network
        self.soft_update(self.local_network, self.traget_network)

    def soft_update(self, local_network, target_network):
        """Soft update the target network."""
