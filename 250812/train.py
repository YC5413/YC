import torch
import numpy as np
import pickle
import time
import gym_super_mario_bros
from utils import create_mario_env
from model import DQNAgent
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import cv2  # 添加OpenCV導入用於影片記錄
import json  # 提前導入 json

# 自定義 JSON 編碼器處理 NumPy 類型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ---------------- Configuration ----------------
CONFIG = {
    'training_mode': True,
    'pretrained': True,
    'double_dqn': True,  
    'stage_specific_epsilon': True,
    'custom_stage_epsilon_max': True,
    'num_episodes': 2000,
    'exploration_max': 0.08,
    'exploration_min': 0.005,
    'exploration_decay': 0.998,
    'max_memory_size': 200000, 
    'batch_size': 32,
    'gamma': 0.99, 
    'learning_rate': 0.00025,
    'save_interval': 500, 
    'env_name': 'SuperMarioBros-v0',
    'model_path': '20250816-001209_episode2000.pt', 
    'checkpoint_dir': 'checkpoints', 
    'stage_initial_epsilons': {},
    'save_replay_buffer': True, 
    'replay_buffer_path': 'replay_buffer.pkl'
}

# 顯示遊戲畫面的函數 - 已不再顯示，改為只記錄
def show_game_screen(env, title="遊戲畫面"):
    """不再顯示遊戲畫面，只返回畫面數據"""
    # 獲取畫面數據但不顯示
    try:
        screen = env.render(mode='rgb_array')
        if screen is None or (isinstance(screen, np.ndarray) and screen.size == 0):
            # 如果無法獲取畫面，創建一個默認的黑色畫面
            screen = np.zeros((240, 256, 3), dtype=np.uint8)
            print("無法獲取遊戲畫面，使用黑色畫面代替")
        elif isinstance(screen, np.ndarray):
            # 確保螢幕尺寸正確
            if screen.shape[0] != 240 or screen.shape[1] != 256:
                screen = cv2.resize(screen, (256, 240))
            # 確保數據類型正確
            if screen.dtype != np.uint8:
                screen = (screen * 255).astype(np.uint8)
    except Exception as e:
        print(f"渲染遊戲畫面時發生錯誤: {e}")
        # 創建一個默認的黑色畫面
        screen = np.zeros((240, 256, 3), dtype=np.uint8)
    return screen

# 初始化影片記錄器
def setup_video_recorder(episode, base_dir="image", session_dir=None):
    """設置影片記錄器"""
    # 創建主目錄
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 使用傳入的會話目錄或創建新的
    if session_dir is None:
        # 創建基於時間的目錄名稱
        session_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    video_dir = os.path.join(base_dir, session_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # 設置影片檔案路徑，使用回合數命名
    video_path = os.path.join(video_dir, f"episode_{episode}.avi")
    
    # 初始化影片寫入器，使用 MJPG 編碼器
    # 對於 Windows 系統更兼容的編碼器
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (240, 256))
    
    # 檢查影片寫入器是否成功打開
    if not video_writer.isOpened():
        print("MJPG 編碼器失敗，嘗試使用 XVID 編碼器...")
        video_path = os.path.join(video_dir, f"episode_{episode}_xvid.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (240, 256))
        
        if not video_writer.isOpened():
            print("XVID 編碼器也失敗，使用不指定編碼器的方式...")
            video_path = os.path.join(video_dir, f"episode_{episode}_raw.avi")
            fourcc = 0  # 自動選擇編碼器
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (240, 256))
    
    return video_writer, video_path

def run():

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # 如果沒有指定模型路徑，則自動尋找最新的檢查點
    if CONFIG['pretrained'] and CONFIG['model_path'] is None:
        checkpoint_files = [f for f in os.listdir(CONFIG['checkpoint_dir']) if f.endswith('.pt') and 'episode' in f]
        if checkpoint_files:
            # 按文件修改時間排序，取最新的
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(CONFIG['checkpoint_dir'], x)), reverse=True)
            CONFIG['model_path'] = checkpoint_files[0]
            print(f"自動選擇最新的檢查點: {CONFIG['model_path']}")
        else:
            CONFIG['pretrained'] = False
            print("未找到任何檢查點，將從頭開始訓練")
    
    model_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['model_path']) if CONFIG['model_path'] else None
    total_rewards_path = os.path.join(CONFIG['checkpoint_dir'], 'total_rewards.pkl')
    original_rewards_path = os.path.join(CONFIG['checkpoint_dir'], 'original_rewards.pkl')
    stage_epsilons_path = os.path.join(CONFIG['checkpoint_dir'], 'stage_epsilons.pkl')
    replay_buffer_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['replay_buffer_path'])

    # 創建基於時間的視頻會話目錄
    video_session_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"創建視頻會話目錄: image/{video_session_dir}")

    # 創建訓練環境
    env = create_mario_env(training=CONFIG['training_mode'])
    obs_shape = env.observation_space.shape
    act_space = env.action_space.n

    # 根據配置決定是否使用預訓練模型
    agent = DQNAgent(
        state_shape=obs_shape,
        n_actions=act_space,
        lr=CONFIG['learning_rate'],
        gamma=CONFIG['gamma'],
        batch_size=CONFIG['batch_size'],
        max_memory_size=CONFIG['max_memory_size'],
        model_path=model_path if CONFIG['pretrained'] else None,  # 如果設定pretrained為True，則使用預訓練模型
        stage_epsilons_path=stage_epsilons_path if CONFIG['pretrained'] else None,
        replay_buffer_path=replay_buffer_path if CONFIG['save_replay_buffer'] and CONFIG['pretrained'] else None
    )

    if CONFIG['stage_specific_epsilon'] and CONFIG['custom_stage_epsilon_max']:
        for stage, init_epsilon in CONFIG['stage_initial_epsilons'].items():
            print(f"Setting initial epsilon for stage {stage} to {init_epsilon}")
            agent.set_stage_epsilon(stage, init_epsilon)

    total_rewards = []
    original_rewards = []
    if CONFIG['training_mode'] and CONFIG['pretrained']:
        try:
            with open(total_rewards_path, 'rb') as f:
                total_rewards = pickle.load(f)
            try:
                with open(original_rewards_path, 'rb') as f:
                    original_rewards = pickle.load(f)
            except FileNotFoundError:
                original_rewards = []
        except FileNotFoundError:
            print("No rewards history found, starting fresh.")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/{current_time}')
    
    print(f"開始訓練! 使用 SIMPLE_MOVEMENT 動作集")
    print(f"總共進行 {CONFIG['num_episodes']} 個回合")
    print(f"訓練模式: {'開啟' if CONFIG['training_mode'] else '關閉'}")
    print(f"預訓練模型: {'使用' if CONFIG['pretrained'] else '不使用'}")
    print(f"雙重 DQN: {'開啟' if CONFIG['double_dqn'] else '關閉'}")
    print(f"階段特定探索率: {'開啟' if CONFIG['stage_specific_epsilon'] else '關閉'}")
    print(f"保存間隔: 每 {CONFIG['save_interval']} 回合")
    print(f"Tensorboard 日誌目錄: runs/{current_time}")
    print("-" * 80)

    for ep in range(CONFIG['num_episodes']):
        state = torch.from_numpy(env.reset()).float()
        total_reward = 0
        original_total_reward = 0
        done = False
        x_pos_prev = 0
        current_stage = None
        
        visited_stages_this_episode = set()
        info = {} 
        
        # 每回合的統計信息
        steps_in_episode = 0
        max_x_pos = 0
        actions_taken = {i: 0 for i in range(act_space)}  # 記錄每個動作的使用次數
        
        # 是否記錄這個回合的影片 - 每100回合記錄一次，以及第1回合
        record_video = (ep + 1) == 1 or (ep + 1) % 100 == 1
        video_writer = None
        frames = []
        
        if record_video:
            print(f"回合 {ep+1} 將被記錄為影片")
            # 先收集幀，稍後再寫入影片
            frames = []

        while not done:
            # 不再顯示畫面，加快訓練速度
            # if not CONFIG['training_mode']:
            #     env.render()
            #     time.sleep(0.01)

            world = info.get('world')
            stage_num = info.get('stage')
            if world is not None and stage_num is not None:
                current_stage = f"{world}-{stage_num}"
                visited_stages_this_episode.add(current_stage)

            stage = current_stage
            if CONFIG['stage_specific_epsilon'] and stage is not None:
                if stage not in agent.stage_epsilons:
                    init_epsilon = CONFIG['stage_initial_epsilons'].get(stage, CONFIG['exploration_max'])
                    agent.set_stage_epsilon(stage, init_epsilon)
                epsilon = agent.get_stage_epsilon(stage, CONFIG['exploration_min'])
            else:
                epsilon = max(CONFIG['exploration_min'], CONFIG['exploration_max'] * (CONFIG['exploration_decay'] ** ep))

            action = agent.select_action(state, epsilon, stage=stage)
            next_obs, reward, done, info = env.step(int(action))
            
            # 統計信息更新
            steps_in_episode += 1
            actions_taken[int(action)] += 1
            x_pos_current = info.get('x_pos', 0)
            max_x_pos = max(max_x_pos, x_pos_current)
            
            # 如果需要記錄影片，捕獲當前幀
            if record_video:
                # 每3步記錄一幀，提供更流暢的影片
                if steps_in_episode % 3 == 0:
                    frame = show_game_screen(env)
                    if frame is not None:
                        # 確保幀的大小正確 (width, height) 是 (240, 256)
                        if frame.shape[0] != 256 or frame.shape[1] != 240:
                            frame = cv2.resize(frame, (240, 256))
                        # 將RGB轉為BGR (OpenCV格式)
                        if frame.shape[2] == 3:  # 確保有色彩通道
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frames.append(frame)

            original_reward = reward
            original_total_reward += original_reward

     
            #if info.get('flag_get'):
            #    reward += 800
            #    print(f"Flag reached!")

            x_pos_current = info.get('x_pos', 0)
            if 'x_pos_prev' in locals():
                x_change = x_pos_current - x_pos_prev
                if x_change > 0:
                    # 增加前進獎勵
                    reward += x_change * 0.1  # 從0.05增加到0.1
                elif x_change < 0:
                    # 增加後退懲罰
                    reward -= 0.2  # 從0.05增加到0.2
                else:
                    # 懲罰原地不動
                    reward -= 0.1
                    
                # 額外獎勵當前位置，鼓勵更遠距離
                if x_pos_current > max_x_pos - 50:  # 接近最遠距離
                    reward += 0.2
            x_pos_prev = x_pos_current
            total_reward += reward
            next_state = torch.from_numpy(next_obs).float()
            agent.memorize(state, action, reward, next_state, done)
            if CONFIG['training_mode']:
                agent.update()
            state = next_state

        for stage in visited_stages_this_episode:
            agent.track_stage_visit(stage)
            
        if CONFIG['stage_specific_epsilon']:
            for stage in visited_stages_this_episode:
                old_eps = agent.get_stage_epsilon(stage, CONFIG['exploration_max'])
                new_eps = agent.decay_stage_epsilon(stage, CONFIG['exploration_min'], CONFIG['exploration_decay'])
                writer.add_scalar(f'Stage_Epsilon/{stage}', new_eps, ep)

        total_rewards.append(total_reward)
        original_rewards.append(original_total_reward)

        writer.add_scalar('Reward/shaped', total_reward, ep)
        writer.add_scalar('Reward/original', original_total_reward, ep)
        writer.add_scalar('Info/x_position', info.get('x_pos', 0), ep)
        writer.add_scalar('Agent/exploration_rate', epsilon, ep)

        shaped_avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) > 0 else 0
        original_avg_reward = np.mean(original_rewards[-100:]) if len(original_rewards) > 0 else 0
        writer.add_scalar('Reward/original_avg', original_avg_reward, ep)

        x_pos = info.get('x_pos', 0)
        
        # 統計每個動作的使用頻率
        action_stats = " | ".join([f"動作{a}: {count}" for a, count in actions_taken.items() if count > 0])
        
        # 只在每100回合結束時輸出詳細資訊
        if (ep + 1) % 100 == 0:
            print(f"回合 {ep+1} 完成 | 階段 {current_stage} | 總步數 {steps_in_episode}")
            print(f"原始獎勵: {original_total_reward:.2f} | 調整獎勵: {total_reward:.2f}")
            print(f"原始平均: {original_avg_reward:.2f} | 調整平均: {shaped_avg_reward:.2f}")
            print(f"最遠位置: {max_x_pos} | 當前位置: {x_pos} | 探索率: {epsilon:.4f}")
            print(f"動作統計: {action_stats}")
            print("-" * 80)
        else:
            # 非100回合的倍數，只顯示簡單進度
            print(f"回合 {ep+1} 完成", end="\r")
        if record_video and frames:
            video_writer, video_path = setup_video_recorder(ep+1, session_dir=video_session_dir)
            print(f"開始將 {len(frames)} 幀寫入影片...")
            
            # 檢查並修復每一幀
            fixed_frames = []
            for i, frame in enumerate(frames):
                # 確保幀是正確的格式和大小
                if frame is None:
                    print(f"幀 {i} 是 None，跳過")
                    continue
                
                # 確保幀大小為 240x256
                if frame.shape[0] != 256 or frame.shape[1] != 240:
                    print(f"幀 {i} 大小錯誤: {frame.shape}，調整為 240x256")
                    frame = cv2.resize(frame, (240, 256))
                
                # 確保幀是 BGR 格式
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(f"幀 {i} 不是 BGR 格式: {frame.shape}，轉換為 BGR")
                    if len(frame.shape) == 2:  # 灰度圖
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # 確保幀數據類型是 uint8
                if frame.dtype != np.uint8:
                    print(f"幀 {i} 數據類型錯誤: {frame.dtype}，轉換為 uint8")
                    frame = (frame * 255).astype(np.uint8)
                
                fixed_frames.append(frame)
            
            # 寫入修復後的幀
            for frame in fixed_frames:
                video_writer.write(frame)
                
            video_writer.release()
            print(f"影片已保存: {video_path}")
            
        # 只在每100回合或發生特殊事件時顯示分隔線
        if (ep + 1) % 100 == 0 or record_video:
            print("-" * 80)
            
        # 每1000回合重置一部分記憶回放緩衝區，避免卡在局部最優解
        if CONFIG['training_mode'] and (ep + 1) % 1000 == 0:
            print("部分重置記憶回放緩衝區以避免局部最優解...")
            agent.reset_memory(partial=True, keep_ratio=0.3)  # 保留30%的記憶

        # 記錄額外的指標到 TensorBoard
        writer.add_scalar('Stats/steps_per_episode', steps_in_episode, ep)
        writer.add_scalar('Stats/max_x_position', max_x_pos, ep)
        for act, count in actions_taken.items():
            writer.add_scalar(f'Actions/action_{act}', count, ep)

        # 在第一回合結束時創建檢查點以測試功能，然後刪除
        if ep == 0:
            current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            test_checkpoint_name = f"{current_datetime}_test.pt"
            test_checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], test_checkpoint_name)
            
            # 創建測試用元數據
            test_meta_data = {
                "episode": int(ep + 1),
                "average_reward": float(shaped_avg_reward),
                "original_average_reward": float(original_avg_reward),
                "max_x_position": int(max_x_pos),
                "steps_in_last_episode": int(steps_in_episode),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            test_checkpoint = {
                'online_net': agent.online_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'stage_epsilons': agent.stage_epsilons,
                'visited_stages': list(agent.visited_stages),
                'current_stage_visits': agent.current_stage_visits,
                'meta': test_meta_data
            }
            
            # 保存測試檢查點
            os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)  # 確保目錄存在
            torch.save(test_checkpoint, test_checkpoint_path)
            print(f"創建測試檢查點: {test_checkpoint_path}")
            
            # 驗證檢查點是否存在
            if os.path.exists(test_checkpoint_path):
                print("檢查點功能正常: 檢查點已成功創建")
                
                # 嘗試載入檢查點以確認其完整性
                try:
                    loaded_checkpoint = torch.load(test_checkpoint_path)
                    print("檢查點功能正常: 檢查點可以成功載入")
                    
                    # 刪除測試檢查點
                    os.remove(test_checkpoint_path)
                    print(f"測試檢查點已刪除")
                except Exception as e:
                    print(f"警告: 檢查點可以創建但無法載入: {e}")
            else:
                print("警告: 檢查點功能可能有問題，無法創建檢查點")

        # 正常的檢查點保存邏輯
        if CONFIG['training_mode'] and (ep + 1) % CONFIG['save_interval'] == 0:
            # 生成日期+回合數格式的檢查點名稱
            current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            checkpoint_name = f"{current_datetime}_episode{ep+1}.pt"
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint_name)
            
            # 創建元數據並確保所有值都是 JSON 可序列化的
            meta_data = {
                "episode": int(ep + 1),
                "average_reward": float(shaped_avg_reward),
                "original_average_reward": float(original_avg_reward),
                "max_x_position": int(max_x_pos),
                "steps_in_last_episode": int(steps_in_episode),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            checkpoint = {
                'online_net': agent.online_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'stage_epsilons': agent.stage_epsilons,
                'visited_stages': list(agent.visited_stages),
                'current_stage_visits': agent.current_stage_visits,
                'meta': meta_data
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"儲存檢查點: {checkpoint_path}")
            
            if CONFIG['save_replay_buffer']:
                agent.save_replay_buffer()
                
            with open(total_rewards_path, "wb") as f:
                pickle.dump(total_rewards, f)
            with open(original_rewards_path, "wb") as f:
                pickle.dump(original_rewards, f)
            meta_path = os.path.join(CONFIG['checkpoint_dir'], "meta.json")
            import json
            
            # 確保 meta_entry 中的 numpy 數據類型被轉換為 Python 原生類型
            meta_entry = checkpoint['meta']
            # 轉換所有可能的 numpy 數據類型
            for key, value in meta_entry.items():
                if hasattr(value, 'item') and callable(value.item):  # 檢查是否為 numpy 數據類型
                    meta_entry[key] = value.item()  # 轉換為 Python 原生類型
            
            try:
                with open(meta_path, "r") as f:
                    meta_list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                meta_list = []
            meta_list.append(meta_entry)
            with open(meta_path, "w") as f:
                json.dump(meta_list, f, indent=2, cls=NumpyEncoder)

    if CONFIG['training_mode']:
        # 生成日期+回合數格式的最終檢查點名稱
        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_name = f"{current_datetime}_episode{CONFIG['num_episodes']}.pt"
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint_name)
        agent.save(checkpoint_path)
        if CONFIG['save_replay_buffer']:
            agent.save_replay_buffer()
            
        with open(total_rewards_path, "wb") as f:
            pickle.dump(total_rewards, f)
        with open(original_rewards_path, "wb") as f:
            pickle.dump(original_rewards, f)
        meta = {
            "episode": CONFIG['num_episodes'],
            "average_reward": shaped_avg_reward,
            "original_average_reward": original_avg_reward,
            "timestamp": datetime.datetime.now().isoformat()
        }
        meta_path = os.path.join(CONFIG['checkpoint_dir'], f"meta_{CONFIG['num_episodes']}.json")
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    writer.close()
    env.close()

if __name__ == "__main__":
    run()