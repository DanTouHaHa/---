import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

# 定义 DQN 网络
class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        Q = self.out(x)
        return Q

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
        return list(self.buffer)[-batch_size:]

    def __len__(self):
        return len(self.buffer)

# 修改 rlalgorithm 类以使用 DQN
class DQN:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.00001, reward_decay=0.9, e_greedy=0.9, memory_size=10000, batch_size=32, target_update=10):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_min = 0.01  # 最小 epsilon 值
        self.epsilon_decay = 0.995  # epsilon 衰减率
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)
        self.display_name = "DQN"
        self.update_counter = 0  # 更新计数器
        
        self.model = DQNNetwork(state_size, hidden_size, action_size)
        self.target_model = DQNNetwork(state_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_update = target_update  # 目标网络的更新频率
        self.update_target_model()

        print("Using DQN ...")

    def choose_action(self, state):
        # self.epsilon = 0.9 - 0.6 * len(self.memory) / self.memory.capacity
        # print(self.epsilon)
        if np.random.uniform() >= self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.model(state)
                action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        # print(action)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # 更新 epsilon
        
        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = zip(*transitions)

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        max_next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * max_next_q_value * (1 - done)

        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def push_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
