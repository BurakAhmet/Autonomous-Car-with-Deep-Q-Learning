
import torch.nn as nn
import torch.optim as optim
import random
import torch
from collections import deque
import numpy as np

# Define the Deep Q-Network (DQN) with Convolutional Neural Network (CNN)
class DQN(nn.Module):
    def __init__(self, action_size):
        """
        Initializes the DQN model architecture.

        Parameters:
        - action_size (int): The number of possible actions in the environment.
        """
        super(DQN, self).__init__()

        # First convolutional layer
        # Input channels: 4 (stacked frames or images)
        # Output channels: 32
        # Kernel size: 8x8
        # Stride: 4
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32,
                               kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        # Input channels: 32
        # Output channels: 64
        # Kernel size: 4x4
        # Stride: 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()

        # Third convolutional layer
        # Input channels: 64
        # Output channels: 64
        # Kernel size: 3x3
        # Stride: 1
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()

        # Compute the size of the flattened feature maps after the convolutional layers
        # Assuming input images of size 84x84, after the conv layers, the feature maps will be 7x7 (due to the kernel sizes and strides)
        # Flattened size = 7 * 7 * 64
        flattened_size = 7 * 7 * 64  # Adjust if input image size changes

        # Fully connected layer
        # Input features: flattened_size + 3 (we concatenate the relative position vector of size 3)
        # Output features: 512
        self.fc1 = nn.Linear(flattened_size + 3, 512)
        self.relu4 = nn.ReLU()

        # Output layer
        # Input features: 512
        # Output features: action_size (Q-values for each possible action)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x, relative_position):
        """
        Defines the forward pass of the network.

        Parameters:
        - x (torch.Tensor): Input image tensor of shape (batch_size, 4, 84, 84)
        - relative_position (torch.Tensor): Relative position tensor of shape (batch_size, 3)

        Returns:
        - torch.Tensor: Output Q-values for each action, shape (batch_size, action_size)
        """
        # Pass through the first convolutional layer and apply ReLU activation
        x = self.conv1(x)
        x = self.relu1(x)

        # Second convolutional layer with ReLU
        x = self.conv2(x)
        x = self.relu2(x)

        # Third convolutional layer with ReLU
        x = self.conv3(x)
        x = self.relu3(x)

        # Flatten the feature maps to a single vector
        x = x.view(x.size(0), -1)  # Shape: (batch_size, flattened_size)

        # Concatenate the relative position information with the flattened feature maps
        x = torch.cat((x, relative_position), dim=1)

        # Pass through the first fully connected layer with ReLU activation
        x = self.fc1(x)
        x = self.relu4(x)

        # Output layer to get Q-values for each action
        x = self.fc2(x)

        return x

# Define the Agent that interacts with the environment and learns from experiences
class Agent:
    def __init__(self, action_size):
        """
        Initializes the agent with hyperparameters and neural networks.

        Parameters:
        - action_size (int): The number of possible actions in the environment.
        """
        self.action_size = action_size

        # Hyperparameters for training
        self.gamma = 0.99             # Discount factor for future rewards
        self.epsilon = 1.0            # Exploration rate (epsilon-greedy)
        self.epsilon_min = 0.05        # Minimum exploration rate
        self.epsilon_decay = 0.990    # Decay rate for epsilon
        self.learning_rate = 0.001    # Learning rate for the optimizer
        self.batch_size = 32          # Mini-batch size for experience replay
        self.memory = deque(maxlen=100000)  # Replay memory buffer

        # Set the device (CPU or GPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the policy network (current model) and the target network
        self.model = DQN(action_size).to(self.device)           # Main network for Q-value prediction
        self.target_model = DQN(action_size).to(self.device)    # Target network for stability
        self.update_target_model()  # Copy weights from the policy network to the target network

        # Optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

    def update_target_model(self):
        """
        Updates the target network weights with the weights from the policy network.
        This is done periodically to stabilize training.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, relative_position, action, reward,
                 next_state, next_relative_position, done):
        """
        Stores experiences in the replay memory.

        Parameters:
        - state (ndarray): Current state (e.g., image frames)
        - relative_position (ndarray): Current relative position vector
        - action (int): Action taken by the agent
        - reward (float): Reward received after taking the action
        - next_state (ndarray): Next state resulting from the action
        - next_relative_position (ndarray): Next relative position vector
        - done (bool): Flag indicating if the episode has terminated
        """
        # Store the experience tuple in memory
        self.memory.append((state, relative_position, action, reward,
                            next_state, next_relative_position, done))

    def act(self, state, relative_position):
        """
        Decides on an action to take based on the current state.

        Parameters:
        - state (ndarray): Current state
        - relative_position (ndarray): Current relative position

        Returns:
        - action (int): The action selected by the agent
        """
        # Implement epsilon-greedy policy for exploration vs exploitation
        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: choose the action with the highest predicted Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            relative_position = torch.FloatTensor(
                relative_position).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Get Q-values from the policy network
                act_values = self.model(state, relative_position)
            # Return the action with the highest Q-value
            return torch.argmax(act_values).item()

    def replay(self):
        """
        Trains the policy network using experiences sampled from the replay memory.
        """
        # Check if there are enough samples in memory to start training
        if len(self.memory) < self.batch_size * 10:
            # Wait until memory has enough samples (arbitrary threshold)
            return

        # Randomly sample a mini-batch of experiences from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract elements from the sampled experiences
        states = np.array([e[0] for e in minibatch], dtype=np.float32)
        relative_positions = np.array(
            [e[1] for e in minibatch], dtype=np.float32)
        actions = torch.LongTensor([e[2] for e in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([e[3] for e in minibatch])
        next_states = np.array([e[4] for e in minibatch], dtype=np.float32)
        next_relative_positions = np.array(
            [e[5] for e in minibatch], dtype=np.float32)
        dones = torch.FloatTensor([e[6] for e in minibatch])

        # Convert numpy arrays to PyTorch tensors and move them to the device
        states = torch.FloatTensor(states).to(self.device)
        relative_positions = torch.FloatTensor(
            relative_positions).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_relative_positions = torch.FloatTensor(
            next_relative_positions).to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute the target Q-values using the target network
        with torch.no_grad():
            # Get the Q-values for the next states from the target network
            next_Q_values = self.target_model(
                next_states, next_relative_positions)
            # Get the maximum Q-value for each next state
            max_next_Q_values = next_Q_values.max(1)[0]
            # Compute the target Q-values
            target_Q_values = rewards + (1 - dones) * \
                self.gamma * max_next_Q_values

        # Compute the predicted Q-values for the actions taken using the policy network
        current_Q_values = self.model(
            states, relative_positions).gather(1, actions).squeeze()

        # Compute the loss between the target Q-values and the predicted Q-values
        loss = self.loss_fn(current_Q_values, target_Q_values)

        # Perform backpropagation and update the network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay the exploration rate epsilon after each training step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay