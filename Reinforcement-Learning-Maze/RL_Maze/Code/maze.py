import time
import tkinter as tk
import numpy as np
import tkinter.messagebox
            
class Maze(tk.Tk, object):
    def __init__(self, agentXY, walls=[], MAZE_H=10, MAZE_W=10, UNIT=20):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.walls = walls
        self.wallblocks = []
        self.UNIT = UNIT
        self.MAZE_H = MAZE_H
        self.MAZE_W = MAZE_W
        self.title('maze')
        self.geometry("{}x{}".format(self.MAZE_W*self.UNIT, self.MAZE_H*self.UNIT))
        if [agentXY[1], agentXY[0]] in walls:
            walls.remove([agentXY[1], agentXY[0]])
        self.build_shape_maze(walls)
        self.start_point = (agentXY[1], agentXY[0])
        self.agent_point = self.start_point
        self.add_agent(agentXY[1], agentXY[0])
        self.add_start(agentXY[1], agentXY[0])

        tkinter.messagebox.showinfo("提示", "请点击选择终点")
        self.canvas.bind("<Button-1>", self.set_point)  # 绑定鼠标点击事件
        self.end_point = None
        self.point_set = False  # 添加一个标志来表示终点是否已设置
        self.best_path = []  # 保存最佳路径
        self.current_path = []  # 保存当前路径
        # self.current_count = []  # 保存当前路径

    def set_point(self, event):
        x, y = event.x // self.UNIT, event.y // self.UNIT
        if not self.end_point:
            if [x, y] not in self.walls:
                self.end_point = (x, y)
                self.add_goal(x, y)
                tkinter.messagebox.showinfo("提示", "终点已设置, 点击OK开始模拟")
                self.point_set = True  # 标记终点已设置
            else:
                tkinter.messagebox.showinfo("错误", "终点不能是墙壁")


    def build_shape_maze(self, walls):
        # 初始化画布
        self.canvas = tk.Canvas(self, bg='black', height=self.MAZE_H * self.UNIT, width=self.MAZE_W * self.UNIT)
        # 绘制网格线
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            self.canvas.create_line(c, 0, c, self.MAZE_H * self.UNIT)
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            self.canvas.create_line(0, r, self.MAZE_W * self.UNIT, r)
        # 添加墙壁
        for x, y in walls:
            self.add_wall(x, y)
        self.canvas.pack()

    def add_wall(self, x, y):
        # 绘制墙壁
        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        wall_center = origin + np.array([self.UNIT * x, self.UNIT * y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - self.UNIT / 2, wall_center[1] - self.UNIT / 2,
            wall_center[0] + self.UNIT / 2, wall_center[1] + self.UNIT / 2,
            fill='white'))

    def add_start(self, x=0, y=0):
        # 标记起始点
        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        agent_center = origin + np.array([self.UNIT * x, self.UNIT * y])
        self.canvas.create_rectangle(
            agent_center[0] - self.UNIT / 2, agent_center[1] - self.UNIT / 2,
            agent_center[0] + self.UNIT / 2, agent_center[1] + self.UNIT / 2,
            fill='yellow')

    def add_goal(self, x=4, y=4):
        # 标记目标点
        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        goal_center = origin + np.array([self.UNIT * x, self.UNIT * y])
        self.goal = self.canvas.create_oval(
            goal_center[0] - self.UNIT / 2, goal_center[1] - self.UNIT / 2,
            goal_center[0] + self.UNIT / 2, goal_center[1] + self.UNIT / 2,
            fill='red')

    def add_agent(self, x=0, y=0):
        # 添加代理
        origin = np.array([self.UNIT / 2, self.UNIT / 2])
        agent_center = origin + np.array([self.UNIT * x, self.UNIT * y])
        self.agent = self.canvas.create_oval(
            agent_center[0] - self.UNIT / 2, agent_center[1] - self.UNIT / 2,
            agent_center[0] + self.UNIT / 2, agent_center[1] + self.UNIT / 2,
            fill='cyan')

    def reset(self, start_point=None, agent_point=None, end_point=None, value=1, resetAgent=True):
        # 重置环境
        self.update()
        time.sleep(0.2)
        if value == 0:
            return self.canvas.coords(self.agent)
        else:
            if resetAgent:
                self.canvas.delete(self.agent)
            if start_point is not None:
                self.start_point = start_point
                self.add_start(*self.start_point)
            if agent_point is not None:
                self.agent_point = agent_point
                self.add_agent(*self.agent_point)
            if end_point is not None:
                self.end_point = end_point
                self.add_goal(*self.end_point)
            self.current_path = []
            return self.canvas.coords(self.agent)

    def computeReward(self, currstate, action, nextstate):
        # 计算奖励
        reverse = False
        if nextstate == self.canvas.coords(self.goal):
            reward = 1
            done = True
            nextstate = 'terminal'
        elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
            reward = -0.3
            done = False
            nextstate = currstate
            reverse = True
        # elif nextstate in self.current_path:
        #     reward = -0.4
        #     done = False
        else:
            reward = -0.1
            end_point = np.array([i for i in self.end_point])
            start_point = np.array([i for i in self.start_point])
            flag = np.linalg.norm(end_point - np.array(self.state_to_xy(currstate)))\
                   - np.linalg.norm(end_point - np.array(self.state_to_xy(nextstate))) 
            global_dist = np.linalg.norm(end_point - np.array(self.state_to_xy(currstate))) \
                        / np.linalg.norm(end_point - start_point) 
            # reward = reward + 0.01 * (1 if flag > 0 else -1)
            reward = reward + 0.2 * (1-global_dist)
            reward = reward + (-0.1 if nextstate in self.current_path else 0)
            # reward = 0.1
            done = False
        return reward, done, reverse

    def step(self, action):
        # 执行一步动作
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # 移动代理
        s_ = self.canvas.coords(self.agent)  # 下一个状态
        reward, done, reverse = self.computeReward(s, action, s_)  # 计算奖励
        if reverse:
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # 如果需要，将代理移回
            s_ = self.canvas.coords(self.agent)
        self.current_path.append(s_)
        return s_, reward, done

    def render(self, sim_speed=0.01):
        # 渲染环境
        time.sleep(sim_speed)
        self.update()
        
    def state_to_xy(self, state):
        assert state[0] + self.UNIT == state[2]
        assert state[1] + self.UNIT == state[3]
        return [state[0]/self.UNIT, state[1]/self.UNIT]

