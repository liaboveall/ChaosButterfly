import numpy as np
import cv2
from hashlib import sha384
from functools import reduce
from operator import xor
import time
import os
import json

class OptimizedNeuralNetwork:
    """
    优化版神经网络类 - 与加密端保持一致
    """
    def __init__(self, sequence_length, hidden_size=10, learning_rate=0.6):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.a = 0.35
        
        # 使用Xavier初始化
        fan_in = 1
        fan_out = hidden_size
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.Ve = np.random.uniform(-limit, limit, (1, hidden_size))
        self.Ws = np.random.uniform(-limit, limit, (hidden_size, 1))
        self.V0e = np.zeros((1, hidden_size))
        self.W0s = np.zeros((1, 1))
        
        self.loss_history = []
    
    def tanh(self, x):
        """优化的tanh函数 - 防止数值溢出"""
        return np.tanh(np.clip(x, -500, 500))
    
    def tanh_derivative(self, x):
        """优化的tanh导数"""
        tanh_x = self.tanh(x)
        return 1.0 - tanh_x**2
    
    def train(self, X, epochs=100, early_stopping_patience=10):
        """优化的训练过程 - 增加早停机制"""
        X = np.array(X).reshape(-1, 1)
        tolerance = 1e-6
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 前向传播
            hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
            output = self.a * (np.dot(hidden, self.Ws) + self.W0s)
            
            # 计算损失
            error = output - X
            total_error = np.mean(np.abs(error))
            self.loss_history.append(total_error)
            
            # 早停检查
            if total_error < best_loss:
                best_loss = total_error
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience or total_error < tolerance:
                break
            
            # 反向传播
            delta_output = self.a * error
            delta_hidden = np.multiply(
                self.tanh_derivative(np.dot(X, self.Ve) + self.V0e),
                np.dot(delta_output, self.Ws.T)
            )
            
            # 更新权重
            self.Ws -= self.learning_rate * np.dot(hidden.T, delta_output) / len(X)
            self.W0s -= self.learning_rate * np.mean(delta_output, axis=0, keepdims=True)
            self.Ve -= self.learning_rate * np.dot(X.T, delta_hidden) / len(X)
            self.V0e -= self.learning_rate * np.mean(delta_hidden, axis=0, keepdims=True)
        
        # 生成最终序列
        hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
        return (self.a * (np.dot(hidden, self.Ws) + self.W0s)).flatten()

class OptimizedLorenzSystem:
    """
    优化版Lorenz混沌系统 - 提高数值稳定性
    """
    def __init__(self, x0, y0, z0):
        self.x = x0
        self.y = y0
        self.z = z0
        self.a = 10.0
        self.b = 40.0
        self.c = 2.5
        self.dt = 0.005
        
        # 添加状态验证
        self._validate_state()
    
    def _validate_state(self):
        """验证系统状态"""
        if not all(np.isfinite([self.x, self.y, self.z])):
            raise ValueError("初始状态包含无效值")
    
    def next_point(self):
        """计算下一个点 - 增加数值稳定性检查"""
        dt = self.dt
        a, b, c = self.a, self.b, self.c
        x, y, z = self.x, self.y, self.z

        # 计算微分
        dx = a * (y - x)
        dy = b * x - x * z + y
        dz = 200 * x * x + 0.01 * np.exp(np.clip(x * y, -700, 700)) - c * z

        # 更新状态
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt
        
        # 防止数值爆炸
        self.x = np.clip(self.x, -1e10, 1e10)
        self.y = np.clip(self.y, -1e10, 1e10)
        self.z = np.clip(self.z, -1e10, 1e10)

        return self.x, self.y, self.z

    def generate_sequences(self, num_points):
        """优化的序列生成"""
        sequences = []
        
        # 预热迭代
        for _ in range(1000):
            self.next_point()
        
        # 批量生成序列
        batch_size = min(1000, num_points)
        for i in range(0, num_points, batch_size):
            batch_points = min(batch_size, num_points - i)
            batch_sequences = []
            
            for _ in range(batch_points):
                x, y, z = self.next_point()
                batch_sequences.append([x, y, z])
            
            sequences.extend(batch_sequences)
        
        sequences = np.array(sequences)
        
        # 优化的归一化
        for i in range(3):
            col = sequences[:, i]
            min_val, max_val = np.min(col), np.max(col)
            if max_val > min_val:
                sequences[:, i] = (col - min_val) / (max_val - min_val)
            else:
                sequences[:, i] = 0.5
        
        return sequences

class OptimizedImageDecryption:
    """
    优化版图像解密类
    """
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.performance_metrics = {}
    
    def inverse_diffuse_image(self, diffused_image, sequence):
        """
        优化的像素扩散逆操作
        """
        start_time = time.time()
        
        M, N = diffused_image.shape
        S, T = self.block_size, self.block_size
        
        # 计算padding
        pad_rows = (S - M % S) % S
        pad_cols = (T - N % T) % T
        padded = np.pad(diffused_image, ((0, pad_rows), (0, pad_cols)), mode='constant')
        
        # 分块
        blocks = []
        for i in range(0, padded.shape[0], S):
            for j in range(0, padded.shape[1], T):
                block = padded[i:i+S, j:j+T].astype(np.uint8)
                blocks.append(block)
        
        num_blocks = len(blocks)
        U = sequence[:num_blocks]
        V = np.argsort(U)
        
        # 生成扩散序列E
        E = np.mod(np.floor(sequence[num_blocks:num_blocks+S*T] * 1e13), 256).astype(np.uint8)
        E = E.reshape(S, T)
        
        # 恢复原始块顺序
        ordered_blocks = [None] * num_blocks
        for i in range(num_blocks):
            ordered_blocks[V[i]] = blocks[i]
        
        # 逆扩散操作
        undiffused_blocks = [None] * num_blocks
        for i in range(num_blocks):
            if i == 0:
                undiffused_blocks[i] = np.bitwise_xor(ordered_blocks[i], E)
            else:
                undiffused_blocks[i] = np.bitwise_xor(ordered_blocks[i], ordered_blocks[i-1])
        
        # 重构图像
        rows = padded.shape[0] // S
        cols = padded.shape[1] // T
        undiffused = np.zeros_like(padded)
        
        for idx, block in enumerate(undiffused_blocks):
            i = (idx // cols) * S
            j = (idx % cols) * T
            undiffused[i:i+S, j:j+T] = block
        
        result = undiffused[:M, :N]
        
        self.performance_metrics['diffusion_time'] = time.time() - start_time
        return result

    def inverse_scramble_image(self, scrambled_image, x_seq, y_seq):
        """
        优化的像素置乱逆操作
        """
        start_time = time.time()
        
        M, N = scrambled_image.shape
        unscrambled = scrambled_image.copy()
        
        # 处理X序列用于行置乱（逆操作）
        x_indices = np.mod(np.floor(x_seq[:M//2] * 1e13), M).astype(int) + 1
        x_indices = np.unique(x_indices)
        missing_nums = np.setdiff1d(np.arange(1, M+1), x_indices)
        x_indices = np.concatenate([x_indices, np.sort(missing_nums)])
        
        # 逆行置乱 - 注意顺序相反
        for i in reversed(range(M//2)):
            if x_indices[i] != x_indices[M-i-1]:
                unscrambled[[x_indices[i]-1, x_indices[M-i-1]-1]] = \
                    unscrambled[[x_indices[M-i-1]-1, x_indices[i]-1]]
        
        # 处理Y序列用于列置乱（逆操作）
        y_indices = np.mod(np.floor(y_seq[:N//2] * 1e13), N).astype(int) + 1
        y_indices = np.unique(y_indices)
        missing_nums = np.setdiff1d(np.arange(1, N+1), y_indices)
        y_indices = np.concatenate([y_indices, np.sort(missing_nums)])
        
        # 逆列置乱 - 注意顺序相反
        for j in reversed(range(N//2)):
            if y_indices[j] != y_indices[N-j-1]:
                unscrambled[:, [y_indices[j]-1, y_indices[N-j-1]-1]] = \
                    unscrambled[:, [y_indices[N-j-1]-1, y_indices[j]-1]]
        
        self.performance_metrics['scrambling_time'] = time.time() - start_time
        return unscrambled

    def validate_decryption(self, original, decrypted):
        """验证解密结果"""
        if original.shape != decrypted.shape:
            return False, "图像尺寸不匹配"
        
        # 计算差异
        diff = np.abs(original.astype(np.float32) - decrypted.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 计算PSNR
        mse = np.mean(diff ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        return {
            'is_perfect': max_diff == 0,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'psnr': psnr,
            'success_rate': np.sum(diff == 0) / diff.size * 100
        }

    def decrypt(self, encrypted_image):
        """
        优化的解密主函数
        """
        decrypt_start = time.time()
        
        if len(encrypted_image.shape) == 3:
            decrypted_channels = []
            for i, channel in enumerate(cv2.split(encrypted_image)):
                print(f"🔄 正在解密第 {i+1} 个颜色通道...")
                decrypted_channel = self.decrypt(channel)
                decrypted_channels.append(decrypted_channel)
            return cv2.merge(decrypted_channels)
        
        # 读取密钥序列
        try:
            sequences = np.load('keys/sequences.npz')
            x_seq = sequences['x_seq']
            y_seq = sequences['y_seq']
            z_seq = sequences['z_seq']
        except FileNotFoundError:
            raise FileNotFoundError("密钥文件不存在，请先运行加密程序")
        
        print("🔓 开始解密过程...")
        
        # 1. 先进行扩散的逆操作
        print("  ➤ 执行扩散逆操作...")
        undiffused = self.inverse_diffuse_image(encrypted_image, z_seq)
        
        # 2. 再进行置乱的逆操作
        print("  ➤ 执行置乱逆操作...")
        decrypted = self.inverse_scramble_image(undiffused, x_seq, y_seq)
        
        total_time = time.time() - decrypt_start
        self.performance_metrics['total_time'] = total_time
        
        print(f"✅ 解密完成！总耗时: {total_time:.3f}秒")
        
        return decrypted.astype(np.uint8)

    def get_performance_report(self):
        """获取性能报告"""
        if not self.performance_metrics:
            return "尚未进行解密操作"
        
        report = "\n📊 解密性能报告:\n"
        report += f"  • 总耗时: {self.performance_metrics.get('total_time', 0):.3f}秒\n"
        report += f"  • 扩散逆操作: {self.performance_metrics.get('diffusion_time', 0):.3f}秒\n"
        report += f"  • 置乱逆操作: {self.performance_metrics.get('scrambling_time', 0):.3f}秒\n"
        
        return report

def main():
    """测试程序"""
    try:
        print("🚀 ChaosButterfly 优化版解密程序")
        print("=" * 50)
        
        # 读取加密图像
        encrypted_img = cv2.imread('output/encrypted.png', cv2.IMREAD_GRAYSCALE)
        if encrypted_img is None:
            raise FileNotFoundError("加密图像文件未找到，请先运行加密程序")

        # 检查密钥文件
        if not os.path.exists('keys/sequences.npz'):
            raise FileNotFoundError("密钥文件未找到，请先运行加密程序生成密钥")

        # 创建输出目录
        os.makedirs("output", exist_ok=True)

        # 解密图像
        decryptor = OptimizedImageDecryption()
        decrypted = decryptor.decrypt(encrypted_img)

        # 保存结果
        cv2.imwrite('output/decrypted_optimized.png', decrypted)
        print(f"💾 解密结果已保存到 output/decrypted_optimized.png")

        # 性能报告
        print(decryptor.get_performance_report())

        # 如果原始图像存在，进行验证
        if os.path.exists('lena.png'):
            original = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
            validation = decryptor.validate_decryption(original, decrypted)
            
            print("\n🔍 解密验证结果:")
            print(f"  • 完美匹配: {'是' if validation['is_perfect'] else '否'}")
            print(f"  • 最大差异: {validation['max_diff']}")
            print(f"  • 平均差异: {validation['mean_diff']:.3f}")
            print(f"  • PSNR值: {validation['psnr']:.2f} dB")
            print(f"  • 成功率: {validation['success_rate']:.2f}%")

    except Exception as e:
        print(f"❌ 出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
