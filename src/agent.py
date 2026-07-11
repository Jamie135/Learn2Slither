import os
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from game import BLOCK_SIZE


class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

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


class Agent:
    def __init__(
        self,
        input_size,
        output_size,
        learning_rate,
        replay_memory_capacity,
        interpolation_steps
    ):
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
        self.interpolation_steps = interpolation_steps
        self.t_steps = 0
        self.recorded_scores = -1
        self.epsilon = -1

    def scan_ray(self, game, dx, dy):
        """Return normalized line-of-sight features for one direction."""
        max_distance = max(1, game.grid_size - 2)

        wall_distance = 0
        body_distance = 0
        green_distance = 0
        red_distance = 0

        x, y = game.snake.x[0], game.snake.y[0]
        distance = 0

        while True:
            distance += 1
            x += dx * BLOCK_SIZE
            y += dy * BLOCK_SIZE

            if (
                x <= 0
                or x >= (game.grid_size - 1) * BLOCK_SIZE
                or y <= 0
                or y >= (game.grid_size - 1) * BLOCK_SIZE
            ):
                wall_distance = distance
                break

            if body_distance == 0:
                for i in range(1, game.snake.length):
                    if game.snake.x[i] == x and game.snake.y[i] == y:
                        body_distance = distance
                        break

            if green_distance == 0:
                for apple in game.green_apples:
                    if apple.x == x and apple.y == y:
                        green_distance = distance
                        break

            if red_distance == 0 and game.red_apple.x == x and game.red_apple.y == y:
                red_distance = distance

        return [
            wall_distance / max_distance,
            body_distance / max_distance if body_distance else 0.0,
            green_distance / max_distance if green_distance else 0.0,
            red_distance / max_distance if red_distance else 0.0,
        ]

    def get_state(self, game):
        """
        Get the ray-based state of the game.

        The state contains:
        - the snake's current direction
        - line-of-sight information in the 4 directions from the head
          for walls, body, green apples, and the red apple
        """
        state = [
            # move direction
            game.snake.direction == "LEFT",
            game.snake.direction == "RIGHT",
            game.snake.direction == "UP",
            game.snake.direction == "DOWN",
        ]

        # Left, right, up, down ray features.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            state.extend(self.scan_ray(game, dx, dy))

        return np.array(state, dtype=np.float32)

    def _valid_actions(self, state):
        """Return action indices that do not immediately reverse direction."""
        moving_left = bool(state[4])
        moving_right = bool(state[5])
        moving_up = bool(state[6])
        moving_down = bool(state[7])

        if moving_right:
            return [0, 1, 3]
        if moving_left:
            return [1, 2, 3]
        if moving_up:
            return [0, 1, 2]
        if moving_down:
            return [0, 2, 3]
        return [0, 1, 2, 3]

    def get_action(self, state, epsilon):
        """
        Get the action for the given state.
        Returning possible choices for the action:
        """
        # .unsqueeze(0) — adds a batch dimension,
        # changing shape from [16] to [1, 16]
        states = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        # disable gradient calculation to speed up inference
        with torch.no_grad():
            # since action size is 4 we will get:
            # [Q(state, action_0), ..., Q(state, action_3)]
            actions = self.local_network(states)
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

    def step(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        minibatch_size,
        gamma,
        interpolation_steps
    ):
        """
        Take a step in the environment.
        Meaning that the snake is making decisions.
        """
        self.memory.push((state, action, reward, next_state, done))
        self.t_steps = (self.t_steps + 1) % 4
        if self.t_steps == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(minibatch_size)
                self.learn(experiences, gamma, interpolation_steps)

    def learn(self, experiences, gamma, interpolation_steps):
        """Learn from the experiences."""
        states, actions, rewards, next_states, dones = experiences
        next_actions = (
            self.traget_network(next_states).detach().max(1)[0].unsqueeze(1)
        )
        # compute the Q-targets using the Bellman equation:
        q_targets = rewards + gamma * next_actions * (1 - dones)
        # compute the Q-values for the current state
        q_values = self.local_network(states).gather(1, actions.long())
        # compute the loss
        loss = F.mse_loss(q_values, q_targets)
        # optimize the local network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update the target network
        self.soft_update(
            self.local_network,
            self.traget_network,
            interpolation_steps
        )

    def soft_update(self, local_network, target_network, interpolation_steps):
        """Ipdate the target network based on interpolation value"""
        # let's say lp = [w1, ..., w3] and tp = [v1, ..., v3]
        # zip will return [(w1, v1), ..., (w3, v3)]
        for local_param, target_param in zip(
            local_network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1 - interpolation_steps) +
                local_param.data * interpolation_steps
            )

    def load_model(self, file_path):
        """Load the model and learning state from a single file."""
        if os.path.exists(file_path):
            checkpoint = torch.load(
                file_path, map_location=self.device, weights_only=False
            )
            if isinstance(checkpoint, dict) \
                    and 'model_state_dict' in checkpoint:
                # New bundled format
                self.local_network.load_state_dict(
                    checkpoint['model_state_dict']
                )
                if 'target_state_dict' in checkpoint:
                    self.traget_network.load_state_dict(
                        checkpoint['target_state_dict']
                    )
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict']
                    )
                self.epsilon = checkpoint.get('epsilon', -1)
                self.recorded_scores = checkpoint.get(
                    'recorded_scores', -1
                )
            else:
                # Legacy format: just state dict
                self.local_network.load_state_dict(checkpoint)
                self.traget_network.load_state_dict(checkpoint)
                self._load_legacy_data()
            print(f"Load trained model from {file_path}")
            if self.recorded_scores != -1:
                print(f"Recorded scores: {self.recorded_scores}")
            if self.epsilon != -1:
                print(f"Epsilon: {self.epsilon}")
        else:
            print(f"Model not found at {file_path}")

    def _load_legacy_data(self):
        """Load data from legacy data.json file (backward compat)."""
        model_path = os.path.join('./models/', 'data.json')
        if os.path.exists(model_path):
            with open(model_path, 'r') as file:
                data = json.load(file)
                if data is not None:
                    self.recorded_scores = data.get(
                        'recorded_scores', -1
                    )
                    self.epsilon = data.get('epsilon', -1)

    def save_model(self, file_path):
        """Save the model and learning state to a single file."""
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        checkpoint = {
            'model_state_dict': self.local_network.state_dict(),
            'target_state_dict': self.traget_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'recorded_scores': self.recorded_scores,
            'epsilon': self.epsilon,
        }
        torch.save(checkpoint, file_path)
