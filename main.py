import os

# ==============================================================================
# [核心补丁] 必须在 import torch 之前设置！
# ==============================================================================
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["OMP_NUM_THREADS"] = "1"
# 尝试禁用 PyTorch 的 JIT 编译缓存，强制它不要尝试编译新内核
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["PYTORCH_JIT"] = "0"

import torch
import numpy as np
from utils.arguments import get_args
from episode import Episode

# ==============================================================================
# [运行时补丁] 深度禁用 JIT
# ==============================================================================
print(f"[*] 应用 RTX 4090 兼容性补丁...")

# 1. 劫持 Python 层的设备检测
torch.cuda.get_device_capability = lambda device=None: (8, 6)

# 2. 禁用所有 JIT 引擎
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if hasattr(torch, "_C"):
    for flag in [
        "_jit_set_nvfuser_enabled",
        "_jit_set_texpr_fuser_enabled",
        "_jit_set_profiling_executor",
        "_jit_set_profiling_mode",
        "_get_graph_executor_optimize",
    ]:
        if hasattr(torch._C, flag):
            getattr(torch._C, flag)(False)

print(f"[*] 补丁应用完成. 当前 PyTorch 版本: {torch.__version__}")
# ==============================================================================

fileName = 'data/matterport_category_mappings.tsv'
if os.path.exists(fileName):
    with open(fileName, 'r') as f:
        text = f.read()
    lines = text.split('\n')[1:]
    items = []
    for l in lines:
        items.append(l.split('    '))
    # 这里似乎原本还有逻辑要处理 items，比如赋值给 episode 中的变量
    # 但原代码中 episode.py 会自己重新读取一遍，所以这里主要是为了检查文件是否存在
else:
    print(f"[Warning] 找不到文件: {fileName}")

def setup_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = get_args()
    setup_seed(args)

    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    navigation_episodes = Episode(args)
    navigation_episodes.start()