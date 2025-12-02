# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
# habitat-lab
from habitat.config.default import get_config as cfg_env
# from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import Env, RLEnv, VectorEnv, make_dataset
from habitat.config import read_write # 必须导入这个
#
from agents.sem_exp import Sem_Exp_Env_Agent
from .utils.vector_env import VectorEnv


def make_env_fn(args, config_env, rank):
    # 适配新版配置结构：config_env.habitat.dataset
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    
    # 使用 read_write 上下文修改配置
    with read_write(config_env):
        config_env.habitat.simulator.scene = dataset.episodes[0].scene_id

    if args.agent == "sem_exp":
        env = Sem_Exp_Env_Agent(args=args, rank=rank,
                                config_env=config_env,
                                dataset=dataset
                                )
    else:
        print('args.agent !!!!! {}'.format(args.agent))
        exit(0)

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".glb.json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext) + 4]
            scenes.append(scene)
    scenes.sort()
    return scenes

def construct_envs_gibson(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="envs/habitat/configs/" + args.task_config)
    
    # [修改] 使用上下文管理器，注释掉 defrost/freeze
    with read_write(basic_config):
        basic_config.habitat.dataset.split = args.split
    
    # basic_config.defrost()  <-- 注释掉
    # basic_config.freeze()   <-- 注释掉

    # [修改] 适配新版属性路径
    scenes = basic_config.habitat.dataset.content_scenes
    
    if "*" in scenes:
        content_dir = os.path.join("data/datasets/objectnav/gibson/v1.1/" + args.split, "content")
        scenes = _get_scenes_from_folder(content_dir)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_path="envs/habitat/configs/" + args.task_config)
        # config_env.defrost() <-- 注释掉

        with read_write(config_env):
            if len(scenes) > 0:
                config_env.habitat.dataset.content_scenes = scenes[
                                                    sum(scene_split_sizes[:i]):
                                                    sum(scene_split_sizes[:i + 1])
                                                    ]
                print("Thread {}: {}".format(i, config_env.habitat.dataset.content_scenes))

            if i < args.num_processes_on_first_gpu:
                gpu_id = 0
            else:
                gpu_id = int((i - args.num_processes_on_first_gpu)
                             // args.num_processes_per_gpu) + args.sim_gpu_id
            gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
            
            # [修改] 适配新版配置路径
            config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

            agent_sensors = ["rgb_sensor", "depth_sensor", "semantic_sensor"]
            config_env.habitat.simulator.agents.agent_0.sensors = agent_sensors

            config_env.habitat.environment.max_episode_steps = args.max_episode_length
            
            # 配置传感器参数
            sim = config_env.habitat.simulator
            sim.rgb_sensor.width = args.env_frame_width
            sim.rgb_sensor.height = args.env_frame_height
            sim.rgb_sensor.hfov = args.hfov
            sim.rgb_sensor.position = [0, args.camera_height, 0]

            sim.depth_sensor.width = args.env_frame_width
            sim.depth_sensor.height = args.env_frame_height
            sim.depth_sensor.hfov = args.hfov
            sim.depth_sensor.min_depth = args.min_depth
            sim.depth_sensor.max_depth = args.max_depth
            sim.depth_sensor.position = [0, args.camera_height, 0]

            sim.semantic_sensor.width = args.env_frame_width
            sim.semantic_sensor.height = args.env_frame_height
            sim.semantic_sensor.hfov = args.hfov
            sim.semantic_sensor.position = [0, args.camera_height, 0]

            sim.turn_angle = args.turn_angle
            config_env.habitat.dataset.split = args.split

        # config_env.freeze() <-- 注释掉
        env_configs.append(config_env)
        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )

    return envs

def construct_envs_hm3d(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="envs/habitat/configs/" + args.task_config)
    
    # [修改] 使用 read_write 上下文修改配置
    with read_write(basic_config):
        basic_config.habitat.dataset.split = args.split
        basic_config.habitat.dataset.data_path = \
            basic_config.habitat.dataset.data_path.replace("v1", args.version)
    
    # [关键修改] 注释掉 freeze
    # basic_config.freeze()

    scenes = args.scenes
    
    # [关键修改] 将 basic_config.DATASET 改为 basic_config.habitat.dataset
    dataset = make_dataset(basic_config.habitat.dataset.type, config=basic_config.habitat.dataset)
    
    if "*" in scenes:
        # [关键修改] 同样修正这里的引用
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    print("Scenes per thread:")
    for i in range(args.num_processes):
        config_env = cfg_env(config_path="envs/habitat/configs/" + args.task_config)

        # 使用 read_write 并在内部进行所有修改
        with read_write(config_env):
            if len(scenes) > 0:
                config_env.habitat.dataset.content_scenes = scenes[
                                                    sum(scene_split_sizes[:i]):
                                                    sum(scene_split_sizes[:i + 1])
                                                    ]
                print("Thread {}: {}".format(i, config_env.habitat.dataset.content_scenes))

            if i < args.num_processes_on_first_gpu:
                gpu_id = 0
            else:
                gpu_id = int((i - args.num_processes_on_first_gpu)
                            // args.num_processes_per_gpu) + args.sim_gpu_id
            gpu_id = min(torch.cuda.device_count() - 1, gpu_id)

            # 修改 GPU ID
            config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

            agent_sensors = ["rgb_sensor", "depth_sensor", "semantic_sensor"] 
            config_env.habitat.simulator.agents.agent_0.sensors = agent_sensors

            # === [补丁开始] 手动构建 sim_sensors ===
            from omegaconf import OmegaConf

            # 创建一个空的字典来存放 sim_sensors
            sim_sensors = {}

            # 把具体的传感器配置填进去
            sim_conf = config_env.habitat.simulator

            # 1. RGB Sensor
            sim_conf.rgb_sensor.uuid = "rgb"
            sim_conf.rgb_sensor.type = "HabitatSimRGBSensor" # [关键修正] 添加类型
            sim_sensors["rgb_sensor"] = sim_conf.rgb_sensor

            # 2. Depth Sensor
            sim_conf.depth_sensor.uuid = "depth"
            sim_conf.depth_sensor.type = "HabitatSimDepthSensor" # [关键修正] 添加类型
            sim_sensors["depth_sensor"] = sim_conf.depth_sensor

            # 3. Semantic Sensor
            sim_conf.semantic_sensor.uuid = "semantic"
            sim_conf.semantic_sensor.type = "HabitatSimSemanticSensor" # [关键修正] 添加类型
            sim_sensors["semantic_sensor"] = sim_conf.semantic_sensor

            # 将这个字典赋值给 agent_0.sim_sensors
            OmegaConf.update(config_env.habitat.simulator.agents.agent_0, "sim_sensors", sim_sensors, force_add=True)
            # === [补丁结束] ===            

            config_env.habitat.environment.max_episode_steps = args.max_episode_length

            # 修改传感器参数 
            sim_conf = config_env.habitat.simulator
            sim_conf.rgb_sensor.width = args.env_frame_width
            sim_conf.rgb_sensor.height = args.env_frame_height
            sim_conf.rgb_sensor.hfov = args.hfov
            sim_conf.rgb_sensor.position = [0, args.camera_height, 0]

            sim_conf.depth_sensor.width = args.env_frame_width
            sim_conf.depth_sensor.height = args.env_frame_height
            sim_conf.depth_sensor.hfov = args.hfov
            sim_conf.depth_sensor.min_depth = args.min_depth
            sim_conf.depth_sensor.max_depth = args.max_depth
            sim_conf.depth_sensor.position = [0, args.camera_height, 0]

            sim_conf.semantic_sensor.width = args.env_frame_width
            sim_conf.semantic_sensor.height = args.env_frame_height
            sim_conf.semantic_sensor.hfov = args.hfov
            sim_conf.semantic_sensor.position = [0, args.camera_height, 0]

            sim_conf.turn_angle = args.turn_angle
            config_env.habitat.dataset.split = args.split

            # 设置场景
            config_env.habitat.simulator.scene = dataset.episodes[0].scene_id

        env_configs.append(config_env)
        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )

    return envs, scenes