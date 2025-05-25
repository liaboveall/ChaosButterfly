#!/usr/bin/env python3
"""
ChaosButterfly 项目管理脚本
统一管理所有功能的入口点
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

class ProjectManager:
    """项目管理器"""
    
    def __init__(self):
        self.project_name = "ChaosButterfly"
        self.version = "2.0.0"
        self.scripts = {
            'encrypt': 'old/encrypt.py',
            'decrypt': 'old/decrypt.py',
            'encrypt_v2': 'src/core/encrypt.py',
            'decrypt_v2': 'src/core/decrypt.py',
            'advanced_encryption': 'src/advanced/encryption.py',
            'test': 'src/analysis/test.py',
            'processing': 'src/utils/processing.py',
            'demo': 'old/demo.py'
        }
        
        self.directories = {
            'keys': 'keys/',
            'output': 'output/',
            'logs': 'logs/'
        }
    
    def print_banner(self):
        """打印项目横幅"""
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                      {self.project_name} v{self.version}                      ║
║              基于神经网络优化的混沌图像加密系统               ║
║                                                              ║
║  🦋 混沌理论 + 🧠 神经网络 + 🔐 图像加密 = 🚀 高安全性        ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_dependencies(self) -> bool:
        """检查依赖项"""
        required_packages = [
            'numpy', 'opencv-python', 'matplotlib', 
            'scipy', 'psutil', 'pycryptodome'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'opencv-python':
                    import cv2
                elif package == 'pycryptodome':
                    import Crypto
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print("❌ 缺少以下依赖包:")
            for pkg in missing_packages:
                print(f"  • {pkg}")
            print(f"\n💡 安装命令: pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有依赖项检查通过")
        return True
    
    def setup_directories(self):
        """设置项目目录"""
        for name, path in self.directories.items():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"📁 创建目录: {path}")
        
        # 创建日志目录
        if not os.path.exists('logs'):
            os.makedirs('logs', exist_ok=True)
    
    def list_available_scripts(self):
        """列出可用脚本"""
        print("\n📋 可用脚本:")
        print("=" * 50)
        
        descriptions = {
            'encrypt': '🔐 基础图像加密',
            'decrypt': '🔓 基础图像解密',
            'encrypt_v2': '⚡ 核心图像加密',
            'decrypt_v2': '⚡ 核心图像解密',
            'advanced_encryption': '🚀 高级加密系统（多模式）',
            'test': '🧪 综合性能与安全性测试',
            'processing': '🛡️ 处理工具（错误处理+内存优化）',
            'demo': '🎬 一键演示程序'
        }
        
        for script_name, script_file in self.scripts.items():
            desc = descriptions.get(script_name, '无描述')
            exists = "✅" if os.path.exists(script_file) else "❌"
            print(f"  {exists} {script_name:20} - {desc}")
    
    def run_script(self, script_name: str, args: List[str] = None) -> bool:
        """运行指定脚本"""
        if script_name not in self.scripts:
            print(f"❌ 未知脚本: {script_name}")
            return False
        
        script_file = self.scripts[script_name]
        if not os.path.exists(script_file):
            print(f"❌ 脚本文件不存在: {script_file}")
            return False
        
        print(f"🚀 运行脚本: {script_name}")
        print("=" * 50)
        
        try:
            # 构建命令
            cmd = [sys.executable, script_file]
            if args:
                cmd.extend(args)
            
            # 运行脚本
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode == 0:
                print(f"\n✅ 脚本 {script_name} 执行成功")
                return True
            else:
                print(f"\n❌ 脚本 {script_name} 执行失败 (退出码: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"❌ 运行脚本时出错: {e}")
            return False
    
    def run_benchmark(self):
        """运行基准测试"""
        print("🏆 开始基准测试...")
        
        # 检查测试图像
        if not os.path.exists('lena.png'):
            print("❌ 缺少测试图像 lena.png")
            return False
          # 运行各种测试
        tests = [
            ('basic_encryption', 'encrypt'),
            ('basic_decryption', 'decrypt'),
            ('optimized_encryption', 'encrypt_v2'),
            ('optimized_decryption', 'decrypt_v2'),
            ('comprehensive_test', 'test')
        ]
        
        results = {}
        
        for test_name, script_name in tests:
            print(f"\n🔄 运行 {test_name}...")
            start_time = datetime.now()
            
            success = self.run_script(script_name)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results[test_name] = {
                'success': success,
                'duration': duration,
                'timestamp': start_time.isoformat()
            }
        
        # 保存基准测试结果
        benchmark_file = 'output/benchmark_results.json'
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        print("\n📊 基准测试报告:")
        print("=" * 50)
        
        total_time = 0
        success_count = 0
        
        for test_name, result in results.items():
            status = "✅" if result['success'] else "❌"
            duration = result['duration']
            total_time += duration
            
            if result['success']:
                success_count += 1
            
            print(f"  {status} {test_name:20} - {duration:.2f}s")
        
        print(f"\n🏁 测试完成:")
        print(f"  • 成功: {success_count}/{len(tests)}")
        print(f"  • 总耗时: {total_time:.2f}s")
        print(f"  • 结果保存到: {benchmark_file}")
        
        return success_count == len(tests)
    
    def clean_output(self):
        """清理输出文件"""
        import shutil
        
        cleanup_dirs = ['output', 'keys', 'logs']
        
        print("🧹 清理输出文件...")
        
        for dir_name in cleanup_dirs:
            if os.path.exists(dir_name):
                try:
                    shutil.rmtree(dir_name)
                    print(f"  ✅ 清理目录: {dir_name}")
                except Exception as e:
                    print(f"  ❌ 清理目录失败 {dir_name}: {e}")
        
        # 重新创建目录
        self.setup_directories()
        print("🏗️  重新创建目录结构")
    
    def show_status(self):
        """显示项目状态"""
        print("\n📊 项目状态:")
        print("=" * 50)
        
        # 检查脚本文件
        print("📜 脚本文件:")
        for script_name, script_file in self.scripts.items():
            exists = os.path.exists(script_file)
            status = "✅" if exists else "❌"
            print(f"  {status} {script_file}")
        
        # 检查目录
        print("\n📁 目录结构:")
        for dir_name, dir_path in self.directories.items():
            exists = os.path.exists(dir_path)
            status = "✅" if exists else "❌"
            if exists:
                file_count = len([f for f in os.listdir(dir_path) 
                                if os.path.isfile(os.path.join(dir_path, f))])
                print(f"  {status} {dir_path} ({file_count} 文件)")
            else:
                print(f"  {status} {dir_path}")
        
        # 检查测试图像
        print("\n🖼️  测试图像:")
        test_images = ['lena.png']
        for img in test_images:
            exists = os.path.exists(img)
            status = "✅" if exists else "❌"
            if exists:
                import cv2
                image = cv2.imread(img)
                if image is not None:
                    print(f"  {status} {img} ({image.shape})")
                else:
                    print(f"  ❌ {img} (无法读取)")
            else:
                print(f"  {status} {img}")
    
    def generate_docs(self):
        """生成项目文档"""
        docs_content = f"""# {self.project_name} v{self.version}

## 项目简介
基于神经网络优化的混沌图像加密系统，结合了Lorenz混沌系统和神经网络技术，提供高安全性的图像加密解决方案。

## 主要特性
- 🦋 **混沌加密**: 基于改进的Lorenz混沌系统
- 🧠 **神经网络优化**: 使用神经网络训练混沌序列
- ⚡ **性能优化**: 多种优化模式（快速/标准/安全）
- 🛡️ **安全增强**: 多层hash、PBKDF2密钥派生
- 🧪 **全面测试**: 性能基准测试和安全性分析
- 📊 **详细报告**: 生成加密质量和性能报告

## 可用脚本

### 基础功能
- `encrypt.py` - 基础图像加密
- `decrypt.py` - 基础图像解密

### 优化版本
- `encrypt_optimized.py` - 优化版图像加密
- `decrypt_optimized.py` - 优化版图像解密

### 高级功能
- `advanced_encryption.py` - 高级加密系统（支持多模式）
- `comprehensive_test.py` - 综合性能与安全性测试
- `robust_processing.py` - 健壮性处理（错误处理+内存优化）

### 辅助工具
- `demo.py` - 一键演示程序
- `manage.py` - 项目管理脚本

## 快速开始

### 1. 安装依赖
```bash
pip install numpy opencv-python matplotlib scipy psutil pycryptodome
```

### 2. 基础使用
```bash
# 运行演示
python manage.py demo

# 基础加密
python manage.py encrypt

# 优化版加密
python manage.py encrypt_optimized

# 综合测试
python manage.py comprehensive_test
```

### 3. 性能基准测试
```bash
python manage.py benchmark
```

## 项目结构
```
{self.project_name}/
├── encrypt.py              # 基础加密
├── decrypt.py              # 基础解密
├── encrypt_optimized.py    # 优化版加密
├── decrypt_optimized.py    # 优化版解密
├── advanced_encryption.py  # 高级加密系统
├── comprehensive_test.py   # 综合测试
├── robust_processing.py    # 健壮性处理
├── demo.py                 # 演示程序
├── manage.py               # 项目管理
├── README.md               # 项目文档
├── lena.png                # 测试图像
├── keys/                   # 密钥文件目录
├── output/                 # 输出文件目录
└── logs/                   # 日志文件目录
```

## 技术细节

### 算法原理
1. **初始值生成**: 使用SHA-384和SHA-256多层hash生成混沌系统初始值
2. **混沌序列**: 改进的Lorenz混沌系统生成高质量随机序列
3. **神经网络优化**: 使用神经网络训练混沌序列，提高随机性
4. **像素置乱**: 基于混沌序列的行列置乱操作
5. **像素扩散**: 分块异或扩散，增强安全性

### 安全特性
- 密钥空间: 2^384 (SHA-384 hash)
- 雪崩效应: 单像素改变导致50%输出变化
- 统计安全性: 加密图像直方图均匀分布
- 相关性破坏: 消除相邻像素相关性

## 性能指标
- 加密速度: 通常 > 10 MB/s
- 内存使用: 支持大图像分块处理
- 安全评分: 综合评分 > 90/100

## 更新日志

### v2.0.0
- 新增高级加密系统
- 添加多种优化模式
- 实现健壮性处理
- 添加综合测试框架
- 改进错误处理和内存管理

### v1.0.0
- 基础加密解密功能
- 神经网络优化
- 性能优化
- 安全性分析

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('README_generated.md', 'w', encoding='utf-8') as f:
            f.write(docs_content)
        
        print("📚 项目文档已生成: README_generated.md")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description=f'ChaosButterfly v2.0.0 项目管理器',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('command', nargs='?', default='help',
                       help='要执行的命令')
    parser.add_argument('--args', nargs='*', default=[],
                       help='传递给脚本的参数')
    
    args = parser.parse_args()
    
    manager = ProjectManager()
    manager.print_banner()
    
    # 检查依赖项
    if not manager.check_dependencies():
        return 1
    
    # 设置目录
    manager.setup_directories()
    
    # 执行命令
    command = args.command.lower()
    
    if command == 'help' or command == '--help':
        print("\n📋 可用命令:")
        print("  help              - 显示此帮助信息")
        print("  status            - 显示项目状态")
        print("  list              - 列出可用脚本")
        print("  benchmark         - 运行性能基准测试")
        print("  clean             - 清理输出文件")
        print("  docs              - 生成项目文档")
        print("  <script_name>     - 运行指定脚本")
        
        manager.list_available_scripts()
        
    elif command == 'status':
        manager.show_status()
        
    elif command == 'list':
        manager.list_available_scripts()
        
    elif command == 'benchmark':
        success = manager.run_benchmark()
        return 0 if success else 1
        
    elif command == 'clean':
        manager.clean_output()
        
    elif command == 'docs':
        manager.generate_docs()
        
    elif command in manager.scripts:
        success = manager.run_script(command, args.args)
        return 0 if success else 1
        
    else:
        print(f"❌ 未知命令: {command}")
        print("💡 使用 'python manage.py help' 查看可用命令")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
