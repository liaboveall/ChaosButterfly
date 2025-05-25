"""
ChaosButterfly v2.0 - 高级加密系统
支持多种加密模式和并行处理
"""

import numpy as np
import cv2
import time
import os
import logging
from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import xor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chaosbutterfly.log'),
        logging.StreamHandler()
    ]
)

class EncryptionMode(Enum):
    """加密模式枚举"""
    FAST = "fast"
    STANDARD = "standard"
    SECURE = "secure"

@dataclass
class EncryptionConfig:
    """加密配置类"""
    mode: EncryptionMode = EncryptionMode.STANDARD
    neural_hidden_size: int = 64
    neural_epochs: int = 50
    block_size: int = 8
    learning_rate: float = 0.01
    patience: int = 10
    validation_split: float = 0.2
    parallel_threads: int = 4
    
    def __post_init__(self):
        """根据模式自动调整参数"""
        if self.mode == EncryptionMode.FAST:
            self.neural_epochs = 20
            self.neural_hidden_size = 32
            self.block_size = 16
        elif self.mode == EncryptionMode.SECURE:
            self.neural_epochs = 200
            self.neural_hidden_size = 128
            self.block_size = 4
            self.patience = 50

class ConfigurableNeuralNetwork:
    """可配置的神经网络"""
    
    def __init__(self, input_size: int, config: EncryptionConfig):
        self.input_size = input_size
        self.hidden_size = config.neural_hidden_size
        self.output_size = input_size
        self.learning_rate = config.learning_rate
        self.epochs = config.neural_epochs
        self.patience = config.patience
        self.validation_split = config.validation_split
        
        # Xavier初始化
        self.weights_ih = np.random.normal(0, np.sqrt(2.0 / (input_size + self.hidden_size)), 
                                          (input_size, self.hidden_size))
        self.weights_ho = np.random.normal(0, np.sqrt(2.0 / (self.hidden_size + self.output_size)), 
                                          (self.hidden_size, self.output_size))
        self.bias_h = np.zeros((1, self.hidden_size))
        self.bias_o = np.zeros((1, self.output_size))
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        
        self.logger = logging.getLogger('ConfigurableNeuralNetwork')
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_ih) + self.bias_h)
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_ho) + self.bias_o)
        return self.output
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        output_error = y - output
        output_delta = output_error * output * (1 - output)
        
        hidden_error = output_delta.dot(self.weights_ho.T)
        hidden_delta = hidden_error * self.hidden * (1 - self.hidden)
        
        self.weights_ho += self.hidden.T.dot(output_delta) * self.learning_rate / m
        self.bias_o += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate / m
        self.weights_ih += X.T.dot(hidden_delta) * self.learning_rate / m
        self.bias_h += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate / m
    
    def adaptive_learning_rate(self, epoch: int, initial_loss: float, current_loss: float) -> float:
        """自适应学习率调整"""
        if epoch == 0:
            return self.learning_rate
        
        # 学习率衰减
        decay_factor = 0.95
        adaptive_rate = self.learning_rate * (decay_factor ** (epoch // 10))
        
        # 如果损失不下降，增加学习率
        if len(self.loss_history) > 1 and current_loss > self.loss_history[-2]['train']:
            adaptive_rate *= 1.1
        
        return max(adaptive_rate, self.learning_rate * 0.1)  # 最小学习率
    
    def train_with_validation(self, X: np.ndarray, validation_split: float = 0.2) -> np.ndarray:
        """带验证的训练"""
        X = np.array(X).reshape(-1, 1)
        
        # 数据分割
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        
        self.logger.info(f"开始神经网络训练 (训练样本: {len(X_train)}, 验证样本: {len(X_val)})")
        
        for epoch in range(self.epochs):
            # 训练
            train_output = self.forward(X_train)
            train_loss = np.mean((X_train - train_output) ** 2)
            self.backward(X_train, X_train, train_output)
            
            # 验证
            val_output = self.forward(X_val)
            val_loss = np.mean((X_val - val_output) ** 2)
            
            # 记录损失
            self.loss_history.append({'train': train_loss, 'val': val_loss})
            
            # 自适应学习率
            if epoch > 0:
                self.learning_rate = self.adaptive_learning_rate(epoch, self.loss_history[0]['train'], train_loss)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"训练在第 {epoch + 1} 轮收敛")
                    break
        
        return self.forward(X).flatten()

class AdvancedLorenzSystem:
    """高级Lorenz混沌系统"""
    
    def __init__(self, x0: float, y0: float, z0: float, config: EncryptionConfig):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.sigma, self.rho, self.beta = 10.0, 28.0, 8.0/3.0
        self.dt = 0.01
        self.config = config
        self.logger = logging.getLogger('AdvancedLorenzSystem')
        
        # 检查初始条件
        if abs(x0) > 50 or abs(y0) > 50 or abs(z0) > 50:
            self.logger.warning("初始状态可能超出典型混沌区域")
    
    def _lorenz_equations(self, state: np.ndarray) -> np.ndarray:
        """Lorenz方程组"""
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])
    
    def _runge_kutta_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """四阶Runge-Kutta数值积分"""
        k1 = self._lorenz_equations(state)
        k2 = self._lorenz_equations(state + 0.5 * dt * k1)
        k3 = self._lorenz_equations(state + 0.5 * dt * k2)
        k4 = self._lorenz_equations(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def generate_sequence_chunk(self, start_state: np.ndarray, length: int) -> np.ndarray:
        """生成序列块"""
        sequence = np.zeros((length, 3))
        current_state = start_state.copy()
        
        for i in range(length):
            sequence[i] = current_state
            current_state = self._runge_kutta_step(current_state, self.dt)
        
        return sequence
    
    def generate_sequences_parallel(self, total_length: int) -> np.ndarray:
        """并行生成混沌序列"""
        chunk_size = total_length // self.config.parallel_threads
        remainder = total_length % self.config.parallel_threads
        
        self.logger.info(f"使用 {self.config.parallel_threads} 线程并行生成 {total_length} 个混沌点")
        
        # 为每个线程准备不同的初始状态
        initial_states = []
        base_state = np.array([self.x0, self.y0, self.z0])
        
        for i in range(self.config.parallel_threads):
            # 轻微扰动初始状态以确保独立性
            perturbed_state = base_state + np.random.normal(0, 0.01, 3)
            initial_states.append(perturbed_state)
        
        # 并行执行
        with ThreadPoolExecutor(max_workers=self.config.parallel_threads) as executor:
            futures = []
            for i in range(self.config.parallel_threads):
                length = chunk_size + (1 if i < remainder else 0)
                future = executor.submit(self.generate_sequence_chunk, initial_states[i], length)
                futures.append(future)
            
            # 收集结果
            sequences = []
            for future in futures:
                sequences.append(future.result())
        
        # 合并序列
        return np.vstack(sequences)

class AdvancedImageEncryption:
    """高级图像加密类"""
    
    def __init__(self, config: EncryptionConfig = None):
        self.config = config or EncryptionConfig()
        self.logger = logging.getLogger('AdvancedImageEncryption')
        
        # 确保必要目录存在
        os.makedirs('keys', exist_ok=True)
        os.makedirs('output', exist_ok=True)
    
    def _generate_initial_values_enhanced(self, image: np.ndarray) -> Tuple[float, float, float]:
        """增强的初始值生成"""
        def enhanced_xor_sum(nums, offset):
            return reduce(xor, nums[offset:offset+16])
        
        # 基础统计特征
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val, max_val = np.min(image), np.max(image)
        
        # 多层哈希
        import hashlib
        combined_data = f"{mean_val:.6f}{std_val:.6f}{min_val}{max_val}".encode()
        hash_sha384 = hashlib.sha384(combined_data).digest()
        hash_sha256 = hashlib.sha256(combined_data).digest()
        
        # 转换为数值
        hash_nums = list(hash_sha384) + list(hash_sha256)
        
        # 基础值
        base_x = (hash_nums[0] + hash_nums[1] * 256) / 65535.0
        base_y = (hash_nums[2] + hash_nums[3] * 256) / 65535.0
        base_z = (hash_nums[4] + hash_nums[5] * 256) / 65535.0
          # 增强计算 - 使用安全的数值计算避免溢出
        x0 = base_x + ((int(enhanced_xor_sum(hash_nums, 0)) + int(mean_val) + int(std_val)) % 256) / 256.0
        y0 = base_y + ((int(enhanced_xor_sum(hash_nums, 16)) + int(min_val) + int(max_val)) % 256) / 256.0  
        z0 = base_z + ((int(enhanced_xor_sum(hash_nums, 32)) + int(mean_val * std_val)) % 256) / 256.0
        
        return float(x0), float(y0), float(z0)
    
    def _encrypt_single_channel(self, image: np.ndarray) -> np.ndarray:
        """单通道加密"""
        # 生成初始值
        x0, y0, z0 = self._generate_initial_values_enhanced(image)
        
        # 生成混沌序列
        lorenz = AdvancedLorenzSystem(x0, y0, z0, self.config)
        M, N = image.shape
        
        # 计算所需序列长度
        scramble_points_x = M  # 行置乱需要M个点
        scramble_points_y = N  # 列置乱需要N个点
        scramble_points = scramble_points_x + scramble_points_y
        diffuse_points = ((M + self.config.block_size - 1) // self.config.block_size) * \
                        ((N + self.config.block_size - 1) // self.config.block_size) + \
                        self.config.block_size * self.config.block_size
        
        # 生成序列
        total_points = scramble_points + diffuse_points
        sequences = lorenz.generate_sequences_parallel(total_points)
        
        # 分离序列
        scramble_seq = sequences[:scramble_points]
        diffuse_seq = sequences[scramble_points:]
          # 神经网络处理 - 修复输入维度问题
        nn_scramble_x = ConfigurableNeuralNetwork(1, self.config)  # 输入维度为1
        nn_scramble_y = ConfigurableNeuralNetwork(1, self.config)  # 输入维度为1
        nn_diffuse = ConfigurableNeuralNetwork(1, self.config)     # 输入维度为1
        
        x_seq = nn_scramble_x.train_with_validation(scramble_seq[:scramble_points_x, 0])
        y_seq = nn_scramble_y.train_with_validation(scramble_seq[scramble_points_x:scramble_points_x+scramble_points_y, 1])
        z_seq = nn_diffuse.train_with_validation(diffuse_seq[:, 2])
        
        # 保存序列和元数据
        self._save_encryption_data(x_seq, y_seq, z_seq, x0, y0, z0)
        
        # 加密操作
        scrambled = self._scramble_image_optimized(image, x_seq, y_seq)
        result = self._diffuse_image_optimized(scrambled, z_seq)
        
        return result.astype(np.uint8)
    
    def _scramble_image_optimized(self, image: np.ndarray, x_seq: np.ndarray, y_seq: np.ndarray) -> np.ndarray:
        """优化的图像置乱"""
        M, N = image.shape
        scrambled = image.copy()
        
        # 预计算索引以提高性能
        x_indices = np.mod(np.floor(x_seq * 1e13), M).astype(int)
        y_indices = np.mod(np.floor(y_seq * 1e13), N).astype(int)
        
        # 确保索引唯一性
        x_indices = self._ensure_unique_indices(x_indices, M)
        y_indices = self._ensure_unique_indices(y_indices, N)
        
        # 矢量化行置乱
        for i in range(M//2):
            idx1, idx2 = x_indices[i], x_indices[M-i-1]
            if idx1 != idx2:
                scrambled[[idx1, idx2]] = scrambled[[idx2, idx1]]
        
        # 矢量化列置乱
        for j in range(N//2):
            idx1, idx2 = y_indices[j], y_indices[N-j-1]
            if idx1 != idx2:
                scrambled[:, [idx1, idx2]] = scrambled[:, [idx2, idx1]]
        
        return scrambled
    
    def _diffuse_image_optimized(self, image: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
        """优化的图像扩散"""
        M, N = image.shape
        diffused = image.copy().astype(int)
        
        # 分块处理
        block_size = self.config.block_size
        seq_idx = 0
        
        for i in range(0, M, block_size):
            for j in range(0, N, block_size):
                # 计算实际块大小
                actual_h = min(block_size, M - i)
                actual_w = min(block_size, N - j)
                
                # 获取扩散密钥
                if seq_idx < len(z_seq):
                    key = int(z_seq[seq_idx] * 255) % 256
                    seq_idx += 1
                else:
                    key = 128  # 默认值
                
                # 扩散操作
                block = diffused[i:i+actual_h, j:j+actual_w]
                diffused[i:i+actual_h, j:j+actual_w] = (block + key) % 256
        
        return diffused.astype(np.uint8)
    
    def _ensure_unique_indices(self, indices: np.ndarray, max_val: int) -> np.ndarray:
        """确保索引唯一性"""
        unique_indices = []
        used = set()
        
        for idx in indices:
            if idx not in used:
                unique_indices.append(idx)
                used.add(idx)
            else:
                # 找到未使用的索引
                for new_idx in range(max_val):
                    if new_idx not in used:
                        unique_indices.append(new_idx)
                        used.add(new_idx)
                        break
        
        return np.array(unique_indices)
    
    def _save_encryption_data(self, x_seq: np.ndarray, y_seq: np.ndarray, z_seq: np.ndarray, 
                             x0: float, y0: float, z0: float):
        """保存加密数据"""
        np.savez_compressed('keys/encryption_data.npz',
                           x_seq=x_seq, y_seq=y_seq, z_seq=z_seq,
                           x0=x0, y0=y0, z0=z0,
                           config_mode=self.config.mode.value)
    
    def encrypt_with_progress(self, image: np.ndarray, progress_callback=None) -> np.ndarray:
        """带进度回调的加密"""
        start_time = time.time()
        
        if progress_callback:
            progress_callback(0)
        
        if len(image.shape) == 3:
            # 彩色图像
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                if progress_callback:
                    progress_callback(int((i / image.shape[2]) * 100))
                result[:, :, i] = self._encrypt_single_channel(image[:, :, i])
        else:
            # 灰度图像
            result = self._encrypt_single_channel(image)
        
        if progress_callback:
            progress_callback(100)
        
        encryption_time = time.time() - start_time
        throughput = (image.nbytes / (1024 * 1024)) / encryption_time
        
        self.logger.info(f"加密完成 - 耗时: {encryption_time:.3f}秒, 吞吐量: {throughput:.2f} MB/s")
        
        return result

def main():
    """主函数 - 演示三种加密模式"""
    print("🚀 ChaosButterfly 高级加密系统")
    print("=" * 50)
    
    # 加载测试图像
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 无法加载测试图像 lena.png")
        return
    
    def progress_callback(percent):
        print(f"\r  正在加密灰度图像 ({percent}%)", end="", flush=True)
    
    # 测试三种模式
    modes = [
        (EncryptionMode.FAST, "encrypted_fast.png"),
        (EncryptionMode.STANDARD, "encrypted_standard.png"),
        (EncryptionMode.SECURE, "encrypted_secure.png")
    ]
    
    for mode, output_file in modes:
        print(f"\n🔄 测试 {mode.value.upper()} 模式...")
        
        config = EncryptionConfig(mode=mode)
        encryptor = AdvancedImageEncryption(config)
        
        start_time = time.time()
        encrypted = encryptor.encrypt_with_progress(img, progress_callback)
        encryption_time = time.time() - start_time
        
        # 保存加密图像
        cv2.imwrite(f'output/{output_file}', encrypted)
        
        throughput = (img.nbytes / (1024 * 1024)) / encryption_time
        
        print(f"\n  ✓ 加密完成: output/{output_file}")
        print(f"  ⏱️  耗时: {encryption_time:.3f}秒")
        print(f"  📊 吞吐量: {throughput:.2f} MB/s")

if __name__ == "__main__":
    main()
