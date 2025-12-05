"""
ZLG CAN Driver Diagnostic Tool
检测 ControlCAN.dll 和相关依赖
"""

import os
import sys
import platform

print("=" * 60)
print("ZLG CAN 驱动诊断工具")
print("=" * 60)

# 1. Python 环境
print(f"\n1. Python 环境:")
print(f"   - 版本: {sys.version.split()[0]}")
print(f"   - 位数: {platform.architecture()[0]}")

# 2. 检查文件
print(f"\n2. 检查必需文件:")
backend_dir = os.path.dirname(os.path.abspath(__file__))

files_to_check = {
    "ControlCAN.dll": "ZLG CAN 驱动主文件",
    "kerneldlls/": "ZLG 内核驱动文件夹（重要）",
}

for file, desc in files_to_check.items():
    path = os.path.join(backend_dir, file)
    exists = os.path.exists(path)
    status = "✓ 存在" if exists else "✗ 缺失"
    print(f"   {status} - {file}: {desc}")
    
    if file == "kerneldlls/" and exists:
        # 列出 kerneldlls 内容
        kernel_files = os.listdir(path)
        print(f"      包含 {len(kernel_files)} 个文件")

# 3. 尝试导入 zlgcan
print(f"\n3. 测试 zlgcan 包:")
try:
    import can
    print(f"   ✓ python-can 已安装")
    
    # 检查是否支持 zlgcan
    if hasattr(can.interfaces, 'BACKENDS'):
        backends = can.interfaces.BACKENDS
        if 'zlgcan' in str(backends).lower():
            print(f"   ✓ zlgcan 接口已注册")
        else:
            print(f"   ⚠ zlgcan 接口未找到")
            print(f"     可用接口: {list(backends.keys())[:5]}...")
    
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")

# 4. 总结
print(f"\n{'=' * 60}")
print("诊断总结:")
print("=" * 60)

if not os.path.exists(os.path.join(backend_dir, "kerneldlls")):
    print("\n⚠️  关键问题：缺少 kerneldlls 文件夹")
    print("\n解决方法：")
    print("1. 打开 USB-CAN Tool 的安装目录")
    print("2. 找到 kerneldlls 文件夹（通常在安装根目录）")
    print("3. 将整个 kerneldlls 文件夹复制到 backend/ 目录")
    print(f"   目标路径: {os.path.join(backend_dir, 'kerneldlls')}")
    print("\n或者，如果您的 USB-CAN Tool 版本使用其他 DLL：")
    print("- 查找 zlgcan.dll 文件")
    print("- 确保所有依赖的 .dll 文件都在 backend/ 或 PATH 中")
else:
    print("\n✓ DLL 文件检查通过")
    print("\n如果仍然无法连接，请检查：")
    print("1. ZLG USB-CAN 设备是否已连接")
    print("2. USB-CAN Tool 是否已关闭（设备独占）")
    print("3. config.py 中的设备类型是否正确")

print("\n" + "=" * 60)
