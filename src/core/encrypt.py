import numpy as np
import cv2
from hashlib import sha384
from functools import reduce
from operator import xor
import time
import os

class OptimizedNeuralNetwork:
    """
    优化版神经网络类 - 增加了早停和批处理
    """
    def __init__(self, sequence_length, hidden_size=10, learning_rate=0.6):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.a = 0.35
        
        # 使用更好的权重初始化（Xavier初始化）
        fan_in = 1
        fan_out = hidden_size
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.Ve = np.random.uniform(-limit, limit, (1, hidden_size))
        self.Ws = np.random.uniform(-limit, limit, (hidden_size, 1))
        self.V0e = np.zeros((1, hidden_size))
        self.W0s = np.zeros((1, 1))
        
        # 训练历史记录
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
            
            # 损失计算
            error = output - X
            current_loss = np.mean(error**2)  # MSE损失
            self.loss_history.append(current_loss)
            
            # 早停检查
            if current_loss < best_loss - tolerance:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
            
            # 反向传播（向量化实现）
            delta_output = 2 * self.a * error / len(X)  # MSE梯度
            delta_hidden = np.multiply(
                self.tanh_derivative(hidden),
                np.dot(delta_output, self.Ws.T)
            )
            
            # 权重更新
            self.Ws -= self.learning_rate * np.dot(hidden.T, delta_output)
            self.W0s -= self.learning_rate * np.mean(delta_output, axis=0, keepdims=True)
            self.Ve -= self.learning_rate * np.dot(X.T, delta_hidden)
            self.V0e -= self.learning_rate * np.mean(delta_hidden, axis=0, keepdims=True)
        
        # 生成最终序列
        hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
        return (self.a * (np.dot(hidden, self.Ws) + self.W0s)).flatten()

class OptimizedLorenzSystem:
    """
    优化版Lorenz系统 - 支持批量生成和数值稳定性检查
    """
    def __init__(self, x0, y0, z0):
        self.reset(x0, y0, z0)
        self.a = 10.0
        self.b = 40.0
        self.c = 2.5
        self.dt = 0.005
        
    def reset(self, x0, y0, z0):
        """重置系统状态"""
        self.x = x0
        self.y = y0
        self.z = z0
        
    def next_point(self):
        """计算下一个点 - 增加数值稳定性检查"""
        dt = self.dt
        a, b, c = self.a, self.b, self.c
        x, y, z = self.x, self.y, self.z

        # 防止数值溢出
        xy_product = np.clip(x * y, -100, 100)
        exp_term = np.clip(0.01 * np.exp(xy_product), 0, 1e6)

        dx = a * (y - x)
        dy = b * x - x * z + y
        dz = 200 * x * x + exp_term - c * z

        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt

        # 防止系统发散
        if abs(self.x) > 1e6 or abs(self.y) > 1e6 or abs(self.z) > 1e6:
            self.x, self.y, self.z = np.random.uniform(-1, 1, 3)

        return self.x, self.y, self.z

    def generate_sequences_batch(self, num_points, warmup=1000):
        """批量生成序列 - 更高效的实现"""
        # 预热
        for _ in range(warmup):
            self.next_point()
        
        # 批量生成
        sequences = np.zeros((num_points, 3))
        for i in range(num_points):
            sequences[i] = self.next_point()
        
        # 向量化归一化
        min_vals = np.min(sequences, axis=0)
        max_vals = np.max(sequences, axis=0)
        
        # 防止除零
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        
        sequences = (sequences - min_vals) / ranges
        return sequences

class SecurityAnalyzer:
    """
    安全性分析工具类
    """
    @staticmethod
    def calculate_entropy(image):
        """计算图像信息熵"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        prob = hist / image.size
        prob = prob[prob > 0]  # 移除零概率
        return -np.sum(prob * np.log2(prob))
    
    @staticmethod
    def calculate_correlation(image1, image2):
        """计算两图像相关性"""
        return np.corrcoef(image1.flatten(), image2.flatten())[0, 1]
    
    @staticmethod
    def calculate_pixel_change_rate(image1, image2):
        """计算像素变化率"""
        return np.sum(image1 != image2) / image1.size * 100
    
    @staticmethod
    def analyze_histogram_uniformity(image):
        """分析直方图均匀性"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        expected = image.size / 256  # 理想均匀分布
        chi_square = np.sum((hist - expected)**2 / expected)
        return chi_square
    
    @staticmethod
    def full_security_analysis(original, encrypted, decrypted=None):
        """完整安全性分析"""
        analysis = {
            'original_entropy': SecurityAnalyzer.calculate_entropy(original),
            'encrypted_entropy': SecurityAnalyzer.calculate_entropy(encrypted),
            'correlation': SecurityAnalyzer.calculate_correlation(original, encrypted),
            'pixel_change_rate': SecurityAnalyzer.calculate_pixel_change_rate(original, encrypted),
            'histogram_uniformity': SecurityAnalyzer.analyze_histogram_uniformity(encrypted)
        }
        
        if decrypted is not None:
            analysis['decryption_mse'] = np.mean((original - decrypted)**2)
            analysis['decryption_success'] = analysis['decryption_mse'] < 1.0
        
        return analysis

class OptimizedImageEncryption:
    """
    优化版图像加密类
    """
    def __init__(self, block_size=8, base_x=0.12, base_y=0.23, base_z=0.34):
        self.block_size = block_size
        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z
        self.security_analyzer = SecurityAnalyzer()
        
    def generate_initial_values(self, image):
        """优化的初始值生成 - 增加更多熵源"""
        # 使用多种哈希增强安全性
        hash_val = sha384(image.tobytes()).digest()
        hash_nums = np.frombuffer(hash_val, dtype=np.uint8)
        
        # 添加图像统计特征作为额外熵源
        mean_val = np.mean(hash_nums)
        std_val = np.std(image.astype(np.float64))
        skew_val = np.mean((image - np.mean(image))**3)
        
        def xor_sum(nums):
            return reduce(xor, nums)
        
        # 增强的初始值计算
        entropy_factor = (std_val + abs(skew_val)) % 256
        x0 = self.base_x + ((xor_sum(hash_nums[:8]) + mean_val + entropy_factor) % 256) / 256
        y0 = self.base_y + ((xor_sum(hash_nums[8:16]) + mean_val + entropy_factor) % 256) / 256
        z0 = self.base_z + ((xor_sum(hash_nums[16:24]) + mean_val + entropy_factor) % 256) / 256
        
        # 保存密钥信息
        os.makedirs("keys", exist_ok=True)
        with open("keys/initial_values.txt", "w") as f:
            f.write(f"x0: {x0}\n")
            f.write(f"y0: {y0}\n")
            f.write(f"z0: {z0}\n")
            f.write(f"entropy_factor: {entropy_factor}\n")
        
        return x0, y0, z0

    def scramble_image_optimized(self, image, x_seq, y_seq):
        """优化的像素置乱 - 使用更安全的置乱算法"""
        M, N = image.shape
        scrambled = image.copy()
        
        # 增强的行置乱
        if len(x_seq) >= M:
            row_indices = np.argsort(x_seq[:M])
            scrambled = scrambled[row_indices, :]
        
        # 增强的列置乱
        if len(y_seq) >= N:
            col_indices = np.argsort(y_seq[:N])
            scrambled = scrambled[:, col_indices]
        
        return scrambled

    def diffuse_image_optimized(self, image, sequence):
        """优化的像素扩散 - 增强安全性"""
        M, N = image.shape
        S, T = self.block_size, self.block_size
        
        # 自适应填充
        pad_rows = (S - M % S) % S
        pad_cols = (T - N % T) % T
        padded = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='reflect')
        
        # 分块处理
        blocks = []
        for i in range(0, padded.shape[0], S):
            for j in range(0, padded.shape[1], T):
                block = padded[i:i+S, j:j+T].astype(np.uint8)
                blocks.append(block)
        
        num_blocks = len(blocks)
        if len(sequence) < num_blocks + S*T:
            raise ValueError("序列长度不足")
        
        # 生成置换序列
        U = sequence[:num_blocks]
        V = np.argsort(U)
        
        # 生成扩散序列
        E = np.mod(np.floor(sequence[num_blocks:num_blocks+S*T] * 1e13), 256).astype(np.uint8)
        E = E.reshape(S, T)
        
        # 增强的扩散操作
        diffused_blocks = [None] * num_blocks
        for i in range(num_blocks):
            original_pos = V[i]
            Q = blocks[original_pos]
            
            if i == 0:
                Q_diffused = np.bitwise_xor(Q, E)
            else:
                # 使用多重异或增强安全性
                prev_block = diffused_blocks[i-1]
                mask = np.mod(np.floor(sequence[num_blocks+S*T+i] * 1e13), 256).astype(np.uint8)
                Q_diffused = np.bitwise_xor(np.bitwise_xor(Q, prev_block), mask)
                
            diffused_blocks[i] = Q_diffused
        
        # 重构图像
        final_blocks = [None] * num_blocks
        for i in range(num_blocks):
            final_blocks[V[i]] = diffused_blocks[i]
        
        rows = (padded.shape[0] // S)
        cols = (padded.shape[1] // T)
        diffused = np.zeros_like(padded)
        
        for idx, block in enumerate(final_blocks):
            i = (idx // cols) * S
            j = (idx % cols) * T
            diffused[i:i+S, j:j+T] = block
        
        return diffused[:M, :N]

    def encrypt(self, image, save_analysis=True):
        """优化的加密主函数"""
        start_time = time.time()
        
        if len(image.shape) == 3:
            encrypted_channels = []
            for channel in cv2.split(image):
                encrypted_channel = self.encrypt(channel, save_analysis=False)
                encrypted_channels.append(encrypted_channel)
            result = cv2.merge(encrypted_channels)
        else:
            # 生成初始值
            x0, y0, z0 = self.generate_initial_values(image)
            
            # 使用优化的Lorenz系统
            lorenz = OptimizedLorenzSystem(x0, y0, z0)
            M, N = image.shape
            
            # 计算所需序列长度
            scramble_points = max(M, N)
            block_count = ((M + self.block_size - 1) // self.block_size) * \
                         ((N + self.block_size - 1) // self.block_size)
            diffuse_points = block_count + self.block_size * self.block_size + block_count
            
            # 生成混沌序列
            total_points = max(scramble_points, diffuse_points)
            chaos_seq = lorenz.generate_sequences_batch(total_points)
            
            # 使用优化的神经网络
            nn = OptimizedNeuralNetwork(sequence_length=total_points)
            
            x_seq = nn.train(chaos_seq[:scramble_points, 0])
            y_seq = nn.train(chaos_seq[:scramble_points, 1])
            z_seq = nn.train(chaos_seq[:diffuse_points, 2])
            
            # 保存序列
            os.makedirs("keys", exist_ok=True)
            np.savez('keys/sequences.npz', 
                    x_seq=x_seq, y_seq=y_seq, z_seq=z_seq,
                    M=M, N=N, block_size=self.block_size)
              # 加密过程
            scrambled = self.scramble_image_optimized(image, x_seq, y_seq)
            result = self.diffuse_image_optimized(scrambled, z_seq)
        
        encryption_time = time.time() - start_time
        
        # 安全性分析
        if save_analysis:
            analysis = self.security_analyzer.full_security_analysis(image, result)
            analysis['encryption_time'] = encryption_time
            
            with open("keys/security_analysis.txt", "w", encoding='utf-8') as f:
                f.write("=== ChaosButterfly 安全性分析报告 ===\n")
                f.write(f"原始图像熵: {analysis['original_entropy']:.4f}\n")
                f.write(f"加密图像熵: {analysis['encrypted_entropy']:.4f}\n")
                f.write(f"相关性: {analysis['correlation']:.6f}\n")
                f.write(f"像素变化率: {analysis['pixel_change_rate']:.2f}%\n")
                f.write(f"直方图均匀性(chi-square): {analysis['histogram_uniformity']:.2f}\n")
                f.write(f"加密耗时: {encryption_time:.4f}秒\n")
        
        return result.astype(np.uint8)

# 测试函数
def test_optimization():
    """测试优化效果"""
    print("🧪 测试优化版加密系统...")
    
    # 读取测试图像
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ 测试图像未找到")
        return
    
    # 创建优化版加密器
    encryptor = OptimizedImageEncryption()
    
    # 加密测试
    start_time = time.time()
    encrypted = encryptor.encrypt(img)
    encryption_time = time.time() - start_time
    
    # 保存结果
    os.makedirs("output", exist_ok=True)
    cv2.imwrite('output/encrypted_optimized.png', encrypted)
    
    print(f"✅ 优化版加密完成，耗时: {encryption_time:.4f}秒")
    print("📊 安全性分析报告已保存到 keys/security_analysis.txt")

if __name__ == "__main__":
    test_optimization()
