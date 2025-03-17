import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 迷宮參數設定
grid_size = 5
target = (grid_size - 1, grid_size - 1)
start = (0, 0)

# 設置障礙物位置
obstacles = [(1, 1), (2, 1), (3, 1), (3, 3), (1, 3)]

# 定義動作與對應效果：0: 上, 1: 下, 2: 左, 3: 右
actions = [0, 1, 2, 3]
action_effects = {
    0: (-1, 0),  # 上
    1: (1, 0),   # 下
    2: (0, -1),  # 左
    3: (0, 1)    # 右
}
action_names = {0: "上", 1: "下", 2: "左", 3: "右"}

# Q-learning 參數 (可調整)
alpha = 0.1      # 學習率
gamma = 0.9      # 折扣因子
epsilon = 0.2    # 探索機率
episodes = 1000  # 訓練回合數
max_steps = 200  # 每回合最大步數

# 初始化 Q 表，每個狀態 (x, y) 皆有四個動作的 Q 值
Q = {}
for i in range(grid_size):
    for j in range(grid_size):
        Q[(i, j)] = np.zeros(len(actions))

def choose_action(state):
    """依據 ε-greedy 策略選擇動作"""
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return int(np.argmax(Q[state]))

def take_action(state, action):
    """根據動作取得下一個狀態，若超出邊界或遇到障礙物則保持原位"""
    effect = action_effects[action]
    new_state = (state[0] + effect[0], state[1] + effect[1])
    
    # 檢查是否超出邊界或遇到障礙物
    if (new_state[0] < 0 or new_state[0] >= grid_size or 
        new_state[1] < 0 or new_state[1] >= grid_size or
        new_state in obstacles):
        new_state = state
        
    return new_state

def get_reward(state, next_state):
    """根據狀態給予獎勵或懲罰"""
    if next_state == target:
        return 100  # 達到目標
    elif next_state == state:  # 遇到障礙物或邊界
        return -5   # 較大的懲罰
    else:
        return -1   # 一般移動的小懲罰

def best_path_from_start():
    """利用當前 Q 表從起點推導出最佳路徑 (避免無限迴圈)"""
    state = start
    path = [state]
    steps = 0
    while state != target and steps < grid_size * grid_size:
        action = int(np.argmax(Q[state]))
        next_state = take_action(state, action)
        # 如果下一步沒有變化或進入循環，則中斷
        if next_state == state or next_state in path:
            break
        path.append(next_state)
        state = next_state
        steps += 1
    return path

def visualize_maze(path=None, show_q_values=False):
    """視覺化迷宮和路徑"""
    plt.figure(figsize=(10, 8))
    
    # 繪製網格
    for i in range(grid_size):
        for j in range(grid_size):
            color = 'white'
            if (i, j) == start:
                color = 'green'
            elif (i, j) == target:
                color = 'red'
            elif (i, j) in obstacles:
                color = 'black'
            elif path and (i, j) in path:
                color = 'yellow'
                
            plt.fill_between([j, j+1], [grid_size-i, grid_size-i], [grid_size-(i+1), grid_size-(i+1)], color=color, alpha=0.6)
            
            # 顯示Q值
            if show_q_values:
                q_values = Q[(i, j)]
                q_text = f"U:{q_values[0]:.1f}\nD:{q_values[1]:.1f}\nL:{q_values[2]:.1f}\nR:{q_values[3]:.1f}"
                plt.text(j+0.5, grid_size-(i+0.5), q_text, ha='center', va='center', fontsize=8)
            
    # 繪製網格線
    for i in range(grid_size+1):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)
    
    # 設置圖例和標題
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.6, label='起點'),
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.6, label='目標'),
        plt.Rectangle((0, 0), 1, 1, color='black', alpha=0.6, label='障礙物')
    ]
    if path:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='yellow', alpha=0.6, label='路徑'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('5x5 迷宮 Q-learning 訓練結果')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, grid_size)
    plt.ylim(0, grid_size)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def visualize_q_values():
    """視覺化 Q 值熱圖"""
    # 創建4x1子圖
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    directions = ["上", "下", "左", "右"]
    
    for a in range(4):
        # 創建一個網格來保存該動作的Q值
        q_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in obstacles:
                    q_grid[i, j] = float('nan')  # 障礙物標記為NaN
                else:
                    q_grid[i, j] = Q[(i, j)][a]
        
        # 繪製熱圖
        sns.heatmap(q_grid, ax=axes[a], cmap="YlGnBu", annot=True, fmt=".1f", 
                    mask=np.isnan(q_grid), cbar=(a == 3))
        axes[a].set_title(f'動作: {directions[a]}')
        axes[a].set_xlabel('列')
        axes[a].set_ylabel('行')
        
    plt.tight_layout()
    plt.suptitle('不同動作的Q值分布', y=1.05, fontsize=16)
    plt.show()

def train_q_learning(visualize_steps=False, visualize_interval=100):
    """訓練Q-learning並可視化過程"""
    # 記錄訓練統計
    steps_per_episode = []
    rewards_per_episode = []
    success_episodes = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        state = start
        total_reward = 0
        
        for step in range(max_steps):
            action = choose_action(state)
            next_state = take_action(state, action)
            reward = get_reward(state, next_state)
            total_reward += reward

            # 更新 Q 值
            best_next = np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            
            state = next_state
            if state == target:
                success_episodes += 1
                break
                
        # 記錄統計數據
        steps_per_episode.append(step + 1)
        rewards_per_episode.append(total_reward)
        
        # 每隔一定回合顯示進度
        if (episode + 1) % visualize_interval == 0:
            success_rate = success_episodes / visualize_interval * 100
            success_episodes = 0
            avg_steps = sum(steps_per_episode[-visualize_interval:]) / visualize_interval
            print(f"Episode: {episode+1}/{episodes}, 平均步數: {avg_steps:.2f}, 成功率: {success_rate:.2f}%")
            
            if visualize_steps:
                # 視覺化當前最佳路徑
                best_path = best_path_from_start()
                visualize_maze(best_path, show_q_values=True)
    
    # 訓練結束，輸出統計數據
    training_time = time.time() - start_time
    print(f"\n訓練完成! 總耗時: {training_time:.2f} 秒")
    print(f"最後10回合平均步數: {sum(steps_per_episode[-10:]) / 10:.2f}")
    
    # 繪製學習曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), steps_per_episode)
    plt.title('每回合步數')
    plt.xlabel('回合')
    plt.ylabel('步數')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), rewards_per_episode)
    plt.title('每回合獎勵')
    plt.xlabel('回合')
    plt.ylabel('總獎勵')
    
    plt.tight_layout()
    plt.show()
    
    return steps_per_episode, rewards_per_episode

def simulate_path(show_details=True):
    """模擬測試，顯示從起點到目標的最佳路徑"""
    print("\n==== 模擬測試：從起點到目標的最佳路徑 ====")
    state = start
    path = [state]
    step = 0
    total_reward = 0
    
    if show_details:
        print("初始狀態:", state)
    
    while state != target and step < grid_size * grid_size:
        action = int(np.argmax(Q[state]))
        next_state = take_action(state, action)
        reward = get_reward(state, next_state)
        total_reward += reward

        if show_details:
            print(f"\n【Step {step+1}】")
            print("當前狀態:", state)
            print("選擇動作:", action, f"({action_names[action]})")
            print("下一個狀態:", next_state)
            print("獎勵:", reward)
            print("狀態", state, "的 Q 值:", Q[state])
        
        path.append(next_state)
        state = next_state
        step += 1
        
        if state == next_state and next_state != target:
            if show_details:
                print("\n無法繼續移動，可能被障礙物阻擋。")
            break

    if state == target:
        print(f"\n成功! 機器人用了 {step} 步到達目標，總獎勵: {total_reward}")
    else:
        print(f"\n失敗! 未能在限定步數內到達目標，總獎勵: {total_reward}")
    
    return path

# 主程序
def run_maze_qlearning(train=True, alpha_val=0.1, gamma_val=0.9, epsilon_val=0.2, episodes_val=1000):
    global alpha, gamma, epsilon, episodes
    
    # 更新參數
    alpha = alpha_val
    gamma = gamma_val
    epsilon = epsilon_val
    episodes = episodes_val
    
    print(f"Q-learning 參數: α={alpha}, γ={gamma}, ε={epsilon}, 回合數={episodes}")
    
    # 視覺化初始迷宮
    print("初始迷宮配置:")
    visualize_maze()
    
    if train:
        # 訓練Q-learning
        print("\n開始訓練 Q-learning...")
        train_q_learning(visualize_steps=False, visualize_interval=100)
    
    # 模擬最佳路徑
    best_path = simulate_path(show_details=True)
    
    # 視覺化最終路徑
    print("\n最終路徑:")
    visualize_maze(best_path, show_q_values=False)
    
    # 視覺化Q值分布
    print("\nQ值分布:")
    visualize_q_values()
    
    return Q, best_path

# 執行程式
if __name__ == "__main__":
    # 運行Q-learning訓練
    run_maze_qlearning(train=True, alpha_val=0.1, gamma_val=0.9, epsilon_val=0.2, episodes_val=1000)
