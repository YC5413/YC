import torch
import os
import sys
import numpy as np
import cv2
import time
import datetime
import gym_super_mario_bros
from utils import create_mario_env
from model import DQNAgent

def check_checkpoint(checkpoint_path):
    """簡單檢查檢查點是否可用

    Args:
        checkpoint_path: 檢查點的路徑
    
    Returns:
        bool: 檢查點是否可用
    """
    print(f"檢查檢查點: {checkpoint_path}")
    
    # 步驟1: 檢查檔案是否存在
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 檢查點檔案不存在!")
        return False
    
    # 步驟2: 嘗試載入檢查點
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("成功載入檢查點檔案")
    except Exception as e:
        print(f"錯誤: 無法載入檢查點檔案: {e}")
        return False
    
    # 步驟3: 檢查檢查點內容
    required_keys = ['online_net', 'target_net']
    missing_keys = [key for key in required_keys if key not in checkpoint]
    
    if missing_keys:
        print(f"錯誤: 檢查點缺少必要的組件: {missing_keys}")
        return False
    else:
        print("檢查點包含所有必要的組件")
    
    # 步驟4: 顯示元數據
    if 'meta' in checkpoint:
        print("檢查點元數據:")
        for key, value in checkpoint['meta'].items():
            print(f"  - {key}: {value}")
    
    print("檢查點驗證成功! 可以正常使用。")
    return True

def test_checkpoint(checkpoint_path, num_frames=300, render=True, save_video=True):
    """測試檢查點的實際效能
    
    Args:
        checkpoint_path: 檢查點的路徑
        num_frames: 測試的幀數
        render: 是否顯示視覺化
        save_video: 是否保存影片
    """
    print(f"測試檢查點: {checkpoint_path}")
    
    # 檢查檔案是否存在
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 檢查點檔案不存在!")
        return False
    
    try:
        # 創建環境
        env = create_mario_env(training=False)
        obs_shape = env.observation_space.shape
        act_space = env.action_space.n
        
        # 創建代理
        agent = DQNAgent(
            state_shape=obs_shape,
            n_actions=act_space,
            lr=0.00025,
            gamma=0.9,
            batch_size=32,
            max_memory_size=200000
        )
        
        # 載入檢查點
        print(f"嘗試載入檢查點...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 載入網絡狀態
        agent.online_net.load_state_dict(checkpoint['online_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        print("成功載入代理模型")
        
        # 設置影片記錄
        video_writer = None
        video_path = None
        
        if save_video:
            # 創建影片輸出目錄
            video_dir = "test_videos"
            os.makedirs(video_dir, exist_ok=True)
            
            # 創建影片檔案
            checkpoint_name = os.path.basename(checkpoint_path).replace(".pt", "")
            video_path = os.path.join(video_dir, f"{checkpoint_name}_test.avi")
            
            # 嘗試使用不同的編解碼器，用簡單的格式避免錯誤
            try:
                # 首先嘗試使用MJPG編碼（最常用）
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (240, 256))
                
                # 檢查視頻寫入器是否成功初始化
                if not video_writer.isOpened():
                    # 嘗試使用原始未壓縮格式
                    fourcc = cv2.VideoWriter_fourcc(*'DIB ')  # 未壓縮RGB
                    video_path = os.path.join(video_dir, f"{checkpoint_name}_test_raw.avi")
                    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (240, 256))
                    print("切換至未壓縮視頻格式...")
            except Exception as e:
                print(f"視頻初始化失敗: {e}")
                video_writer = None
        
        # 開始測試
        print(f"開始測試 {num_frames} 幀...")
        state = torch.from_numpy(env.reset()).float()
        frames = []
        total_reward = 0
        max_pos = 0
        
        for i in range(num_frames):
            # 選擇動作
            action = agent.select_action(state, epsilon=0.05)  # 使用低探索率
            
            # 執行動作
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # 更新最大位置
            if 'x_pos' in info:
                max_pos = max(max_pos, info['x_pos'])
            
            # 獲取當前畫面
            screen = env.render(mode='rgb_array')
            
            # 處理畫面
            if screen is not None:
                # 將RGB轉為BGR
                screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                
                # 添加資訊文字
                cv2.putText(screen_bgr, f"Action: {action}", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(screen_bgr, f"Reward: {total_reward:.1f}", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(screen_bgr, f"Position: {info.get('x_pos', 0)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 保存幀
                frames.append(screen_bgr)
                
                # 寫入影片
                if video_writer is not None and video_writer.isOpened():
                    try:
                        video_writer.write(screen_bgr)
                    except Exception as e:
                        # 忽略視頻寫入錯誤，但確保不會中斷測試
                        pass
                
                # 顯示畫面
                if render:
                    cv2.imshow('Test', screen_bgr)
                    cv2.waitKey(1)
            
            # 如果遊戲結束，重置環境
            if done:
                print(f"遊戲結束於第 {i+1} 幀, 總獎勵: {total_reward:.1f}, 最遠位置: {max_pos}")
                state = torch.from_numpy(env.reset()).float()
            else:
                state = torch.from_numpy(next_state).float()
        
        # 關閉影片寫入器
        if video_writer is not None:
            video_writer.release()
            print(f"測試影片已保存至: {video_path}")
        
        # 關閉環境
        env.close()
        
        # 關閉顯示視窗
        if render:
            cv2.destroyAllWindows()
        
        print(f"測試完成: 總獎勵: {total_reward:.1f}, 最遠位置: {max_pos}")
        return True
    
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("簡易檢查點工具")
    print("="*50)
    
    # 檢查是否提供了檢查點路徑
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # 如果沒有提供路徑，則尋找檢查點
        default_dir = "checkpoints"
        if not os.path.exists(default_dir):
            print(f"錯誤: 找不到默認檢查點目錄 '{default_dir}'")
            print("請輸入檢查點路徑:")
            checkpoint_path = input().strip()
        else:
            # 找到目錄中的所有檢查點
            checkpoints = [f for f in os.listdir(default_dir) if f.endswith('.pt')]
            
            if not checkpoints:
                print(f"錯誤: 在 '{default_dir}' 中找不到檢查點")
                print("請輸入檢查點路徑:")
                checkpoint_path = input().strip()
            else:
                # 依照修改時間排序，最新的放前面
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(default_dir, x)), reverse=True)
                
                # 顯示找到的檢查點，並添加"最後修改時間"信息
                print(f"找到以下檢查點:")
                for i, cp in enumerate(checkpoints):
                    # 獲取最後修改時間
                    mod_time = os.path.getmtime(os.path.join(default_dir, cp))
                    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                    # 從檔名中提取回合數
                    episode_info = ""
                    if "episode" in cp:
                        try:
                            episode_info = cp.split("episode")[1].split(".")[0]
                            episode_info = f"回合: {episode_info}"
                        except:
                            pass
                    print(f"{i+1}. {cp} ({mod_time_str}) {episode_info}")
                
                # 請用戶選擇
                print("\n請選擇要使用的檢查點編號:")
                try:
                    choice = int(input().strip())
                    if 1 <= choice <= len(checkpoints):
                        checkpoint_path = os.path.join(default_dir, checkpoints[choice-1])
                    else:
                        print("無效的選擇!")
                        return
                except ValueError:
                    print("請輸入有效的數字!")
                    return
    
    # 檢查檢查點
    print("\n步驟 1: 檢查檢查點是否可用")
    print("-"*50)
    valid = check_checkpoint(checkpoint_path)
    
    if not valid:
        print("檢查點無法使用，無法繼續測試。")
        return
    
    # 詢問是否要測試
    print("\n是否要測試這個檢查點的實際效能? (Y/N)")
    response = input().strip().upper()
    
    if response == 'Y':
        print("\n步驟 2: 測試檢查點效能")
        print("-"*50)
        print("要測試多少幀? (默認: 300)")
        
        try:
            frames = input().strip()
            frames = int(frames) if frames else 300
            
            test_checkpoint(checkpoint_path, num_frames=frames)
        except ValueError:
            print("請輸入有效的數字! 使用默認值300幀。")
            test_checkpoint(checkpoint_path, num_frames=300)
    
    print("\n檢查點工具執行完畢!")

if __name__ == "__main__":
    main()
