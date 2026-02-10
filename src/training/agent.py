import os
import json
import torch
import random
import numpy as np
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

    def get_state(self, game):
        """
        Get the state of the game describing:
        - the snake position
        - its potential moves
        - its potential danger
        - its potential reward
        """
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

    def load_model(self, file_name='model.pth'):
        """Load the model."""
        file_path = os.path.join('./models/', file_name)
        if os.path.exists(file_path):
            self.local_network.load_state_dict(torch.load(file_path))
            self.load_data()
            print(f"Model loaded from {file_path}")
        else:
            print(f"Model not found at {file_path}")

    def load_data(self):
        """Retrieve the data."""
        file_name = 'data.json'
        model_path = os.path.join('./models/', file_name)
        if os.path.exists(model_path):
            with open(model_path, 'r') as file:
                data = json.load(file)
                if data is not None:
                    self.recorded_scores = data['recorded_scores']
                    self.epsilon = data['epsilon']
                    print(f"Recorded scores: {self.recorded_scores}")
                    print(f"Epsilon: {self.epsilon}")
                else:
                    print(f"Data is None at {model_path}")
        else:
            print(f"Data not found at {model_path}")
        return self.recorded_scores, self.epsilon

    def save_model(self, file_name='model.pth'):
        """Save the model."""
        if not os.path.exists('./models/'):
            os.makedirs('./models/')
        file_path = os.path.join('./models/', file_name)
        torch.save(self.local_network.state_dict(), file_path)
        print(f"Model saved to {file_path}")
        self.save_data(self.recorded_scores, self.epsilon)

    def save_data(self, recorded_scores, epsilon):
        """Save the data."""
        file_name = 'data.json'
        if not os.path.exists('./models/'):
            os.makedirs('./models/')
        path = os.path.join('./models/', file_name)
        data = {
            'recorded_scores': recorded_scores,
            'epsilon': epsilon
        }
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {path}")
