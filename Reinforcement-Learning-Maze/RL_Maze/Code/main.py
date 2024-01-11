import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from maze import Maze
from q_learning import Q_Leanring
from dqn import DQN

DEBUG = 1
def debug(debuglevel, msg, **kwargs):
    # 如果debug等级小于或等于DEBUG，打印消息
    # 可以通过kwargs中的'printNow'来控制是否立即打印消息
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs and kwargs['printNow']:
            print(msg) 
        else:
            print(msg)
            
def update(env, RL, data, episodes=50, refresh_rate=10):
    # 更新环境和强化学习算法
    global_reward = np.zeros(episodes)
    data['global_reward'] = global_reward
    max_reward = float('-inf')

    for episode in range(episodes):
        t = 0
        state = env.reset(start_point=env.start_point, agent_point=env.agent_point, end_point=env.end_point, value=0) if episode == 0 else env.reset(start_point=env.start_point, agent_point=env.agent_point, end_point=env.end_point)
        debug(2, 'state(ep:{},t:{})={}'.format(episode, t, state))
        step_counter = 0
        if RL.display_name == "DQN":
            action = RL.choose_action(env.state_to_xy(state))
        elif RL.display_name == "Q-Learning":
            action = RL.choose_action(str(state))

        while True:
            if episode > 100 or (step_counter % refresh_rate == 0):
                env.render(sim_speed)
            step_counter += 1

            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2, 'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2, 'reward_{}= total return_t ={} Mean50={}'.format(reward, global_reward[episode], np.mean(global_reward[-50:])))

            if RL.display_name == "DQN":
                # 存储到记忆回放中
                RL.push_memory(env.state_to_xy(state), action, reward, env.state_to_xy(state_), done)
                # 学习过程
                RL.learn()
                action = RL.choose_action(env.state_to_xy(state_))
                state = state_
            elif RL.display_name == "Q-Learning":
                state, action = RL.learn(str(state), action, reward, str(state_))
                
            if done:
                break
            else:
                t += 1

        if done and global_reward[episode] > max_reward:
            max_reward = global_reward[episode]
            env.best_path = list(env.current_path)

        debug(1, "({}) Episode {}: Length={} Total return = {} ".format(RL.display_name, episode, t, global_reward[episode]), printNow=(episode % printEveryNth == 0))
        if episode >= 100:
            debug(1, "Median100={} Variance100={}".format(np.median(global_reward[episode - 100:episode]), np.var(global_reward[episode - 100:episode])), printNow=(episode % printEveryNth == 0))
    
    plot_best_path(env)
    print('game over -- Algorithm {} completed')

def plot_rewards(experiments):
    # 绘制不同实验的奖励曲线图
    # 每个实验包含环境(env)、学习算法(RL)和数据(data)
    color_list = ['blue', 'green', 'red', 'black', 'magenta']
    label_list = []
    for i, (env, RL, data) in enumerate(experiments):
        x_values = range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values = data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i], label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()

def plot_best_path(env):
    # 绘制最佳路径
    for _, point in enumerate(env.best_path[0:-1]):
        env.canvas.create_oval(point[0], point[1], point[0] + env.UNIT, point[1] + env.UNIT, fill='cyan')
    env.update()
    
def gen_MazeWall_fromFig(img_path):
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)
    grid_size = 8
    height, width = img_array.shape
    img_array = img_array[48:height-40, 0+50:width-50]
    height, width = img_array.shape
    result = np.zeros((height // grid_size, width // grid_size))

    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            grid = img_array[i:i+grid_size, j:j+grid_size]
            avg_value = np.mean(grid)
            if avg_value < 128 / 2.0:
                result[i // grid_size, j // grid_size] = 0
            else:
                result[i // grid_size, j // grid_size] = 1
    result[48][63]=1
    result[44][63]=0
    return result


if __name__ == "__main__":
    
    # 设置迷宫与起点
    startXY=[44, 63]
    result = gen_MazeWall_fromFig('task_maze.jpg')
    np.set_printoptions(threshold=np.inf)
    x, y = np.where(result == 1)
    wall_shape = []
    for i in range(len(x)):
        wall_shape.append([y[i], x[i]])
        
    # 设置模拟参数
    sim_speed = 0.05
    showRender = False
    episodes = 100
    renderEveryNth = 10000
    printEveryNth = 1
    do_plot_rewards = True
    refresh_rate = 100

    # 解析命令行参数
    if(len(sys.argv) > 1):
        episodes = int(sys.argv[1])
    if(len(sys.argv) > 2):
        showRender = sys.argv[2] in ['true', 'True', 'T', 't']
    if(len(sys.argv) > 3):
        datafile = sys.argv[3]
    experiments = []

    # 实验：使用Q-Learning算法
    env1 = Maze(startXY, wall_shape, MAZE_H=result.shape[0], MAZE_W=result.shape[1], UNIT=10)
    RL1 = Q_Leanring(actions=list(range(env1.n_actions)))
    # RL1 = DQN(state_size=2, action_size=4)
    data1 = {}
    while not env1.point_set:
        env1.update()  # 更新tkinter事件循环
    env1.after(100, update(env1, RL1, data1, episodes, refresh_rate))
    env1.mainloop()
    experiments.append((env1, RL1, data1))

    print("experiments complete")

    # 打印实验结果
    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']), np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))

    # 绘制奖励曲线图
    if(do_plot_rewards):
        plot_rewards(experiments)
