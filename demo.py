#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🦋 ChaosButterfly - 演示脚本
基于神经网络优化混沌系统的图像加密器

使用方法:
    python demo.py
"""

import os
import cv2
import numpy as np
from encrypt import ImageEncryption
from decrypt import ImageDecryption

def print_banner():
    """打印欢迎横幅"""
    banner = """
    🦋 ================================ 🦋
       欢迎使用 ChaosButterfly 图像加密器
       基于神经网络优化的混沌加密系统
    🦋 ================================ 🦋
    """
    print(banner)

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查测试图像
    if not os.path.exists('lena.png'):
        print("❌ 测试图像 lena.png 未找到")
        return False
    
    # 创建必要目录
    os.makedirs("output", exist_ok=True)
    os.makedirs("keys", exist_ok=True)
    
    print("✅ 环境检查通过")
    return True

def run_encryption():
    """运行加密流程"""
    print("\n🔐 开始图像加密...")
    
    try:
        # 读取原始图像
        img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("无法读取测试图像")
        
        print(f"📋 原始图像尺寸: {img.shape}")
        
        # 执行加密
        encryptor = ImageEncryption()
        encrypted = encryptor.encrypt(img)
        
        # 保存加密结果
        cv2.imwrite('output/encrypted.png', encrypted)
        
        # 计算统计信息
        original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        encrypted_hist = cv2.calcHist([encrypted], [0], None, [256], [0, 256])
        
        original_entropy = -np.sum((original_hist/img.size) * np.log2(original_hist/img.size + 1e-10))
        encrypted_entropy = -np.sum((encrypted_hist/img.size) * np.log2(encrypted_hist/img.size + 1e-10))
        
        print(f"📊 原始图像信息熵: {original_entropy:.4f}")
        print(f"📊 加密图像信息熵: {encrypted_entropy:.4f}")
        print(f"📈 熵值提升: {((encrypted_entropy/original_entropy - 1) * 100):.2f}%")
        print("✅ 加密完成！文件保存至 output/encrypted.png")
        
        return True
        
    except Exception as e:
        print(f"❌ 加密失败: {e}")
        return False

def run_decryption():
    """运行解密流程"""
    print("\n🔓 开始图像解密...")
    
    try:
        # 检查必要文件
        if not os.path.exists('output/encrypted.png'):
            raise FileNotFoundError("加密图像文件不存在，请先运行加密")
        
        if not os.path.exists('keys/sequences.npz'):
            raise FileNotFoundError("密钥文件不存在，请先运行加密")
        
        # 读取加密图像
        encrypted_img = cv2.imread('output/encrypted.png', cv2.IMREAD_GRAYSCALE)
        
        # 执行解密
        decryptor = ImageDecryption()
        decrypted = decryptor.decrypt(encrypted_img)
        
        # 保存解密结果
        cv2.imwrite('output/decrypted.png', decrypted)
        
        print("✅ 解密完成！文件保存至 output/decrypted.png")
        
        # 验证解密正确性
        original = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if original is not None:
            mse = np.mean((original - decrypted) ** 2)
            if mse < 1.0:  # 允许微小误差
                print("🎉 解密验证通过！图像完美恢复")
            else:
                print(f"⚠️  解密验证: MSE = {mse:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 解密失败: {e}")
        return False

def show_results():
    """展示结果统计"""
    print("\n📈 加密效果分析:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        original = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        encrypted = cv2.imread('output/encrypted.png', cv2.IMREAD_GRAYSCALE)
        
        if original is not None and encrypted is not None:
            # 计算相关性
            correlation = np.corrcoef(original.flatten(), encrypted.flatten())[0, 1]
            
            # 计算像素变化率
            changed_pixels = np.sum(original != encrypted)
            change_rate = (changed_pixels / original.size) * 100
            
            print(f"🔗 原图与密图相关性: {correlation:.6f}")
            print(f"🔄 像素变化率: {change_rate:.2f}%")
            print(f"📁 输出文件位置:")
            print(f"   • 加密图像: output/encrypted.png")
            print(f"   • 解密图像: output/decrypted.png")
            print(f"   • 密钥文件: keys/sequences.npz")
            
    except Exception as e:
        print(f"❌ 统计分析失败: {e}")

def main():
    """主程序"""
    print_banner()
    
    if not check_environment():
        return
    
    # 执行加密
    if not run_encryption():
        return
    
    # 执行解密
    if not run_decryption():
        return
    
    # 展示结果
    show_results()
    
    print("\n🎊 ChaosButterfly 演示完成！")
    print("💡 提示: 查看 output/ 目录中的结果图像")

if __name__ == "__main__":
    main()
