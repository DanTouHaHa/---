import numpy as np
import pandas as pd

class Q_Leanring:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = "Q-Learning"
        print("use Q-Learning ...")

    # 根据观察到的状态选择下一个动作
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() >= self.epsilon:
            state_action_values = self.q_table.loc[observation, :]
            action = np.random.choice(state_action_values[state_action_values == np.max(state_action_values)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    # 学习方法
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_target = r + self.gamma * self.q_table.loc[s_, :].max() if s_ != 'terminal' else r
        self.q_table.loc[s, a] += self.lr * (q_target - self.q_table.loc[s, a])
        return s_, self.choose_action(str(s_))

    # 检查状态是否存在于Q表中，如果不存在则添加
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = pd.concat([self.q_table, pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state).to_frame().T], ignore_index=False)

