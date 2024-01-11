from maze_env import Maze
from RL_brainsample_PI import rlalgorithm as rlalg0
from RL_brain_q_learning import rlalgorithm as rlalg1
from RL_brain_sarsa import rlalgorithm as rlalg2
from RL_brain_expected_sarsa import rlalgorithm as rlalg4
from RL_brain_sarsa_lambda import rlalgorithm as rlalg5
import sys
import matplotlib.pyplot as plt
import threading
from PIL import Image
import numpy as np
import tkinter as tk

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 


def plot_rewards(experiments):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
#    plt.axis([0, 1100, 0, 1100000])
    plt.show()

def update(env, RL, data, episodes=50, refresh_rate=10):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    max_reward = float('-inf')

    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(start_point=env.start_point, end_point=env.end_point, value=0)
        else:
            state = env.reset(start_point=env.start_point, end_point=env.end_point)

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
        step_counter = 0
        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            #if(t<5000 and (showRender or (episode % renderEveryNth)==0)):
            #if(showRender or (episode % renderEveryNth)==0):
            if (episode > 100 or (step_counter % refresh_rate == 0)):
                env.render(sim_speed)
            step_counter += 1

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))


            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        if done:
            # 检查并更新最佳路径
            if global_reward[episode] > max_reward:
                max_reward = global_reward[episode]
                env.best_path = list(env.current_path) # 深拷贝当前路径
        #plot_best_path(env)
            #break
        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    plot_best_path(env)
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    #env.destroy()


def plot_best_path(env):

    for index, point in enumerate(env.best_path):
        #if index == 0:
        #    continue # 跳过第一个点，因为它是起点
        env.canvas.create_oval(
            point[0] - env.UNIT/4, point[1] - env.UNIT/4,
            point[0] + env.UNIT/4, point[1] + env.UNIT/4,
            fill='cyan') # 你可以选择不同的颜色或形状
    env.update()



if __name__ == "__main__":

    # 打开图片
    img = Image.open('task_maze.jpg').convert('L')

    # 将图片转化为numpy数组
    img_array = np.array(img)

    # 确定每个方格的大小
    grid_size = 8  # 例如，每个方格是10x10像素

    # 获取图片的宽度和高度
    height, width = img_array.shape

    img_array = img_array[48:height-40, 0+50:width-50]
    height, width = img_array.shape

    # 创建一个新的二维数组来存储结果
    result = np.zeros((height // grid_size, width // grid_size))

    # 遍历每个方格
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            # 获取当前方格的像素值
            grid = img_array[i:i+grid_size, j:j+grid_size]
            
            # 计算方格的平均像素值
            avg_value = np.mean(grid)
            
            # 根据平均像素值判断方格的颜色
            if avg_value > 128/2.0:
                result[i // grid_size, j // grid_size] = 1  # 白色方格
            else:
                result[i // grid_size, j // grid_size] = 0  # 黑色方格

    np.set_printoptions(threshold=np.inf)
    with open("1.txt", 'w') as f:
        print(result, file=f)

    x,y = np.where(result==1)
    wall_shape = []
    for i in range(len(x)):
        wall_shape.append([y[i],x[i]])
    pits=[]
    
    # ---------------------------------------------------------------------------
    sim_speed = 0.05

    # #Example Short Fast for Debugging
    # showRender=True
    # episodes=30
    # renderEveryNth=5
    # printEveryNth=1
    # do_plot_rewards=True

    #Exmaple Full Run, you may need to run longer
    showRender=False
    episodes=10
    renderEveryNth=10000
    printEveryNth=1
    do_plot_rewards=True
    refresh_rate = 10

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    #All Tasks
    agentXY=[1,1]
    goalXY=[1,5]

    # #Task 1
    # wall_shape=np.array([[7,7],[4,6]])
    # pits=np.array([[6,3],[2,6]])

    # # Task 2
    # wall_shape=np.array([[5,2],[4,2],[3,2],[3,3],[3,4],[3,5],[3,6],[4,6],[5,6]])
    # pits=[]

    #Task 3
    # wall_shape=np.array([[7,4],[7,3],[6,3],[6,2],[5,2],[4,2],[3,2],[3,3],[3,4],[3,5],[3,6],[4,6],[5,6]])
    # pits=np.array([[1,3],[0,5], [7,7]])

    experiments = []

    # # alg0 (Aynsc)
    # env0 = Maze(agentXY,goalXY,wall_shape, pits)
    # RL0 = rlalg0(actions=list(range(env0.n_actions)))
    # data0={}
    # env0.after(10, update(env0, RL0, data0, episodes))
    # env0.mainloop()
    # experiments = [(env0,RL0, data0)]

    # alg1 (Q-Learning)
    env1 = Maze(agentXY, goalXY, wall_shape, pits, MAZE_H=result.shape[0], MAZE_W=result.shape[1], UNIT=10)
    RL1 = rlalg1(actions=list(range(env1.n_actions)))
    data1={}
    while not env1.point_set:
        env1.update()  # 更新tkinter事件循环
    env1.after(100, update(env1, RL1, data1, episodes, refresh_rate))
    env1.mainloop()
    experiments.append((env1, RL1, data1))

    print("experiments complete")

    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))


    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments)


    #Not implemented yet
    #if(do_save_data):
    #    for env, RL, data in experiments:
    #        saveData(env,RL,data)
"""
    # alg2 (SARSA)
    env2 = Maze(agentXY,goalXY,wall_shape,pits)
    RL2 = rlalg2(actions=list(range(env2.n_actions)))
    data2={}
    env2.after(10, update(env2, RL2, data2, episodes))
    env2.mainloop()
    experiments.append((env2,RL2, data2))

    

    # alg4 (Expected Sarsa)
    env4 = Maze(agentXY,goalXY,wall_shape,pits)
    RL4 = rlalg4(actions=list(range(env4.n_actions)))
    data4={}
    env4.after(10, update(env4, RL4, data4, episodes))
    env4.mainloop()
    experiments.append((env4,RL4, data4))

    # alg5 (SARSA_lambda)
    env5 = Maze(agentXY,goalXY,wall_shape,pits)
    RL5 = rlalg5(actions=list(range(env5.n_actions)))
    data5={}
    env5.after(10, update(env5, RL5, data5, episodes))
    env5.mainloop()
    experiments.append((env5,RL5, data5))
"""

