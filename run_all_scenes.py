import os
import subprocess
import glob

def run_all():
    # HM3D 验证集场景的存放路径
    scenes_dir = "data/scene_datasets/hm3d/val"
    
    if not os.path.exists(scenes_dir):
        print(f"Error: 找不到路径 {scenes_dir}")
        return

    # 获取所有场景文件夹
    # 文件夹名通常格式为: 00800-TEEsavR23oF
    scene_folders = sorted(os.listdir(scenes_dir))
    
    print(f"找到 {len(scene_folders)} 个场景，准备开始测试...")

    for i, folder in enumerate(scene_folders):
        # 提取场景 ID (例如从 00800-TEEsavR23oF 提取出 TEEsavR23oF)
        if "-" in folder:
            scene_id = folder.split("-")[-1]
        else:
            scene_id = folder

        print(f"\n[{i+1}/{len(scene_folders)}] 正在测试场景: {scene_id} ...")
        
        # 构造命令
        # 注意：这里我们不加 -efw/-efh，使用默认的高分辨率
        cmd = [
            "python3", "main.py",
            "-d", "Results/",          # 结果保存路径
            "--skip_times", "0",       # 每次都从 Episode 0 开始记
            "--num_eval_episodes", "5",# 每个场景跑 5 轮
            "--scenes", scene_id       # 指定当前场景 ID
        ]

        try:
            # 执行命令并等待完成
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"场景 {scene_id} 运行出错: {e}")
        except KeyboardInterrupt:
            print("\n用户中断，停止脚本。")
            break

    print("\n所有场景测试完成！")

if __name__ == "__main__":
    run_all()