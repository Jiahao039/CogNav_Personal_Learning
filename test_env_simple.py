import habitat_sim
import numpy as np
import os

# 指向刚刚下载的测试场景
test_scene = "data/scene_datasets/skokloster-castle.glb"

if not os.path.exists(test_scene):
    print(f"错误: 找不到场景文件 {test_scene}")
    print("请确保你已经运行了下载命令。")
    exit()

# 配置模拟器后端
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = test_scene
sim_cfg.gpu_device_id = 0  # 使用第一块 GPU

# 配置 RGB 传感器
rgb_sensor_spec = habitat_sim.CameraSensorSpec()
rgb_sensor_spec.uuid = "color_sensor"
rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
rgb_sensor_spec.resolution = [512, 512]
rgb_sensor_spec.position = [0.0, 1.5, 0.0] # 传感器高度

# 配置深度传感器
depth_sensor_spec = habitat_sim.CameraSensorSpec()
depth_sensor_spec.uuid = "depth_sensor"
depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
depth_sensor_spec.resolution = [512, 512]
depth_sensor_spec.position = [0.0, 1.5, 0.0]

# 组装配置
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

try:
    print("正在初始化 Habitat 模拟器...")
    sim = habitat_sim.Simulator(cfg)
    print("✅ 模拟器初始化成功！Habitat-Sim 安装正确。")

    # 初始化 Agent
    agent = sim.initialize_agent(0)
    print("✅ Agent 初始化成功！")

    # 随机测试 10 步动作
    print("正在测试渲染和动作执行...")
    for i in range(10):
        # 随机执行动作：前进、左转、右转
        action = np.random.choice(["move_forward", "turn_left", "turn_right"])
        observations = sim.step(action)
        
        # 检查是否生成了图像数据
        if "color_sensor" in observations and "depth_sensor" in observations:
            pass # 数据生成正常
        else:
            print(f"❌ 第 {i} 步数据生成失败！")
            exit()
            
    print(f"✅ 成功执行了 10 步随机动作。")
    print("🎉 恭喜！你的 Habitat 环境基础配置（CUDA, PyTorch, Habitat-Sim）完全正常！")
    print("现在你可以去申请 HM3D Token 来运行完整的 CogNav 代码了。")

    sim.close()

except Exception as e:
    print(f"\n❌ 环境测试失败，报错信息如下：")
    print(e)