import sys
import os

print("="*60)
print("�� 开始 CogNav 环境全面体检...")
print("="*60)

# 1. 检查 PyTorch 和 CUDA (最关键)
print(f"\n[1/5] 检查 PyTorch & 显卡 (核心引擎)...")
try:
    import torch
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"   CUDA 版本 (PyTorch视角): {torch.version.cuda}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        count = torch.cuda.device_count()
        print(f"✅ 显卡识别成功: {device_name} (共 {count} 张)")
        
        # 测试 Tensor 计算
        x = torch.rand(5, 3).cuda()
        print(f"✅ GPU Tensor 测试: 通过 (能够分配显存)")
    else:
        print("❌ 致命错误: PyTorch 无法识别显卡！")
except Exception as e:
    print(f"❌ PyTorch 检查失败: {e}")

# 2. 检查 Habitat (最难装的部分)
print(f"\n[2/5] 检查 Habitat 仿真器...")
try:
    import habitat
    import habitat_sim
    print(f"✅ Habitat-Lab 版本: {habitat.__version__}")
    print(f"✅ Habitat-Sim 版本: {habitat_sim.__version__}")
    print("   Habitat 依赖加载正常。")
except ImportError as e:
    print(f"❌ Habitat 导入失败: {e}")
except Exception as e:
    print(f"❌ Habitat 未知错误: {e}")

# 3. 检查 PyTorch3D (3D 感知)
print(f"\n[3/5] 检查 PyTorch3D...")
try:
    import pytorch3d
    print(f"✅ PyTorch3D 版本: {pytorch3d.__version__}")
except ImportError:
    print("❌ PyTorch3D 未安装 (请检查是否执行了 pip install pytorch3d -f ...)")
except Exception as e:
    print(f"❌ PyTorch3D 错误: {e}")

# 4. 检查 ChamferDist (我们故意跳过的)
print(f"\n[4/5] 检查 ChamferDist (预期可能失败)...")
try:
    import chamferdist
    print(f"🎉 奇迹！ChamferDist 竟然安装成功了！版本: {chamferdist.__version__}")
except ImportError:
    print("⚠️  ChamferDist 未安装 (正常现象，我们在 requirements.txt 中跳过了它)")
    print("   -> 提示: 如果代码报错缺这个库，请搜索代码并将相关引用替换为 pytorch3d.loss.chamfer_distance")

# 5. 检查其他杂项依赖
print(f"\n[5/5] 检查其他常用库...")
pkgs = {
    "numpy": "NumPy",
    "cv2": "OpenCV",
    "open3d": "Open3D",
    "gradio": "Gradio",
    "tyro": "Tyro",
    "clip": "CLIP (OpenCLIP)",
    "openai": "OpenAI"
}

for pkg, name in pkgs.items():
    try:
        if pkg == "clip":
            import open_clip as clip # CogNav使用的是open-clip-torch
        else:
            __import__(pkg)
        print(f"✅ {name}: OK")
    except ImportError:
        print(f"❌ {name}: 未找到 (可能需要 pip install)")
    except Exception as e:
        print(f"⚠️ {name}: 加载异常 ({e})")

print("\n" + "="*60)
print("体检结束。如果是全绿(或仅 ChamferDist 警告)，则环境配置完美！")
print("="*60)
