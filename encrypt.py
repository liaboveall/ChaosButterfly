import numpy as np
import cv2
from hashlib import sha384
from functools import reduce
from operator import xor

class NeuralNetwork:
    """
    用于训练混沌序列的神经网络类
    严格按照论文1.2节实现
    """
    def __init__(self, sequence_length, hidden_size=10, learning_rate=0.6):
        self.learning_rate = learning_rate  # 论文中的ψ参数
        self.hidden_size = hidden_size      # 隐藏层神经元数量ncc
        self.sequence_length = sequence_length
        self.a = 0.35                       # 论文中的比例系数a
        
        # 初始化权重和偏置
        self.Ve = np.random.randn(1, hidden_size) * 0.01
        self.Ws = np.random.randn(hidden_size, 1) * 0.01
        self.V0e = np.zeros((1, hidden_size))
        self.W0s = np.zeros((1, 1))
    
    def tanh(self, x):
        """双曲正切激活函数 - 用于隐藏层"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """双曲正切导数"""
        return 1.0 - np.tanh(x)**2
    
    def train(self, X, epochs=100):
        """按论文方程(2)-(7)实现训练过程"""
        X = np.array(X).reshape(-1, 1)
        tolerance = 1e-6
        prev_error = float('inf')
        
        for _ in range(epochs):
            total_error = 0
            
            # 前向传播 - 方程(2)(3)
            hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
            # 注意：输出层使用线性激活 g(ξ)=aξ
            output = self.a * (np.dot(hidden, self.Ws) + self.W0s)
            
            # 误差计算 - 方程(4)
            error = output - X
            total_error = np.mean(np.abs(error))
            
            # 反向传播 - 方程(5)
            delta_output = self.a * error
            delta_hidden = np.multiply(
                self.tanh_derivative(hidden),
                np.dot(delta_output, self.Ws.T)
            )
            
            # 更新权重和偏置 - 方程(6)(7)
            self.Ws -= self.learning_rate * np.dot(hidden.T, delta_output) / len(X)
            self.W0s -= self.learning_rate * np.mean(delta_output, axis=0, keepdims=True)
            self.Ve -= self.learning_rate * np.dot(X.T, delta_hidden) / len(X)
            self.V0e -= self.learning_rate * np.mean(delta_hidden, axis=0, keepdims=True)
            
            # 收敛检查
            if abs(total_error - prev_error) < tolerance:
                break
            prev_error = total_error
        
        # 生成最终序列
        hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
        return (self.a * (np.dot(hidden, self.Ws) + self.W0s)).flatten()

class LorenzSystem:
    """
    改进的Lorenz混沌系统
    严格按照论文1.1节方程(1)实现
    """
    def __init__(self, x0, y0, z0):
        self.x = x0
        self.y = y0
        self.z = z0
        self.a = 10.0
        self.b = 40.0
        self.c = 2.5
        self.dt = 0.005
    
    def next_point(self):
        """计算下一个点"""
        dt = self.dt
        a, b, c = self.a, self.b, self.c
        x, y, z = self.x, self.y, self.z

        dx = a * (y - x)
        dy = b * x - x * z + y
        dz = 200 * x * x + 0.01 * np.exp(x * y) - c * z

        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt

        return self.x, self.y, self.z

    def generate_sequences(self, num_points):
        """生成混沌序列"""
        sequences = []
        
        # 预热迭代以达到混沌状态
        for _ in range(1000):
            self.next_point()
        
        # 记录序列范围以便归一化
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        
        for _ in range(num_points):
            x, y, z = self.next_point()
            sequences.append([x, y, z])
            
            x_min, x_max = min(x_min, x), max(x_max, x)
            y_min, y_max = min(y_min, y), max(y_max, y)
            z_min, z_max = min(z_min, z), max(z_max, z)
        
        sequences = np.array(sequences)
        
        # 归一化到[0, 1]区间
        for i in range(3):
            min_val = [x_min, y_min, z_min][i]
            max_val = [x_max, y_max, z_max][i]
            if max_val > min_val:
                sequences[:, i] = (sequences[:, i] - min_val) / (max_val - min_val)
            else:
                sequences[:, i] = 0.5
        
        return sequences

class ImageEncryption:
    """
    图像加密类
    严格按照论文第2节实现
    """
    def __init__(self, block_size=8, base_x=0.12, base_y=0.23, base_z=0.34):
        self.block_size = block_size
        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z
    
    def generate_initial_values(self, image):
        """
        按照论文2.1节方程(8)(9)生成初始值
        """
        hash_val = sha384(image.tobytes()).digest()
        hash_nums = np.frombuffer(hash_val, dtype=np.uint8)
        mean_val = np.mean(hash_nums)
        
        # 使用异或运算处理hash值
        def xor_sum(nums):
            return reduce(xor, nums)
        
        # 按照论文方程(9)计算初始值
        x0 = self.base_x + ((xor_sum(hash_nums[:8]) + mean_val) % 256) / 256
        y0 = self.base_y + ((xor_sum(hash_nums[8:16]) + mean_val) % 256) / 256
        z0 = self.base_z + ((xor_sum(hash_nums[16:24]) + mean_val) % 256) / 256
          # 将初始值记录到keys目录中
        import os
        os.makedirs("keys", exist_ok=True)
        with open("keys/initial_values.txt", "w") as f:
            f.write(f"x0: {x0}\n")
            f.write(f"y0: {y0}\n")
            f.write(f"z0: {z0}\n")

        return x0, y0, z0

    def scramble_image(self, image, x_seq, y_seq):
        """
        按照论文2.2节方程(10)(11)实现像素置乱
        """
        M, N = image.shape
        scrambled = image.copy()
        
        # 处理X序列用于行置乱
        x_indices = np.mod(np.floor(x_seq[:M//2] * 1e13), M).astype(int) + 1
        x_indices = np.unique(x_indices)  # 去重
        missing_nums = np.setdiff1d(np.arange(1, M+1), x_indices)
        x_indices = np.concatenate([x_indices, np.sort(missing_nums)])
        
        # 行置乱
        for i in range(M//2):
            if x_indices[i] != x_indices[M-i-1]:
                scrambled[[x_indices[i]-1, x_indices[M-i-1]-1]] = \
                    scrambled[[x_indices[M-i-1]-1, x_indices[i]-1]]
        
        # 处理Y序列用于列置乱
        y_indices = np.mod(np.floor(y_seq[:N//2] * 1e13), N).astype(int) + 1
        y_indices = np.unique(y_indices)  # 去重
        missing_nums = np.setdiff1d(np.arange(1, N+1), y_indices)
        y_indices = np.concatenate([y_indices, np.sort(missing_nums)])
        
        # 列置乱
        for j in range(N//2):
            if y_indices[j] != y_indices[N-j-1]:
                scrambled[:, [y_indices[j]-1, y_indices[N-j-1]-1]] = \
                    scrambled[:, [y_indices[N-j-1]-1, y_indices[j]-1]]
        
        return scrambled

    def diffuse_image(self, image, sequence):
        """
        按照论文2.3节方程(12)(13)实现像素扩散
        """
        M, N = image.shape
        S, T = self.block_size, self.block_size
        
        # 分块处理
        blocks = []
        pad_rows = (S - M % S) % S
        pad_cols = (T - N % T) % T
        padded = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='constant')
        
        # 将图像分成S×T的块
        for i in range(0, padded.shape[0], S):
            for j in range(0, padded.shape[1], T):
                block = padded[i:i+S, j:j+T].astype(np.uint8)
                blocks.append(block)
        
        # 生成U序列并获取V序列
        num_blocks = len(blocks)
        U = sequence[:num_blocks]
        V = np.argsort(U)
        
        # 生成扩散序列E
        E = np.mod(np.floor(sequence[num_blocks:num_blocks+S*T] * 1e13), 256).astype(np.uint8)
        E = E.reshape(S, T)
        
        # 对每个块进行扩散操作
        diffused_blocks = [None] * num_blocks
        for i in range(num_blocks):
            original_pos = V[i]  # 块在原序列中的位置
            Q = blocks[original_pos]
            
            if i == 0:
                Q_diffused = np.bitwise_xor(Q, E)
            else:
                prev_block = diffused_blocks[i-1]
                Q_diffused = np.bitwise_xor(Q, prev_block)
                
            diffused_blocks[i] = Q_diffused
        
        # 将块放回原始位置
        final_blocks = [None] * num_blocks
        for i in range(num_blocks):
            final_blocks[V[i]] = diffused_blocks[i]
        
        # 重构图像
        rows = (padded.shape[0] // S)
        cols = (padded.shape[1] // T)
        diffused = np.zeros_like(padded)
        
        for idx, block in enumerate(final_blocks):
            i = (idx // cols) * S
            j = (idx % cols) * T
            diffused[i:i+S, j:j+T] = block
        
        # 移除padding
        diffused = diffused[:M, :N]
        return diffused

    def encrypt(self, image):
        """
        加密主函数
        """
        if len(image.shape) == 3:
            encrypted_channels = []
            for channel in cv2.split(image):
                encrypted_channel = self.encrypt(channel)
                encrypted_channels.append(encrypted_channel)
            return cv2.merge(encrypted_channels)
        
        # 生成初始值
        x0, y0, z0 = self.generate_initial_values(image)
        
        # 生成混沌序列
        lorenz = LorenzSystem(x0, y0, z0)
        M, N = image.shape
        
        # 分别生成置乱和扩散所需的序列
        scramble_points = max(M//2, N//2)  # 行列置乱所需点数
        diffuse_points = (M//self.block_size) * (N//self.block_size) + self.block_size * self.block_size
        
        # 生成置乱序列
        scramble_seq = lorenz.generate_sequences(scramble_points)
        x_seq, y_seq = scramble_seq[:, 0], scramble_seq[:, 1]
        
        # 重新初始化Lorenz系统生成扩散序列
        lorenz = LorenzSystem(x0, y0, z0)
        diffuse_seq = lorenz.generate_sequences(diffuse_points)
        
        # 使用神经网络处理序列
        nn_scramble = NeuralNetwork(sequence_length=len(scramble_seq))
        nn_diffuse = NeuralNetwork(sequence_length=len(diffuse_seq))
        
        x_seq = nn_scramble.train(x_seq)
        y_seq = nn_scramble.train(y_seq)
        z_seq = nn_diffuse.train(diffuse_seq[:, 2])        # 将序列保存到keys目录中
        import os
        os.makedirs("keys", exist_ok=True)
        np.savez('keys/sequences.npz', x_seq=x_seq, y_seq=y_seq, z_seq=z_seq)
        
        # 置乱和扩散
        scrambled = self.scramble_image(image, x_seq, y_seq)
        encrypted = self.diffuse_image(scrambled, z_seq)
        
        return encrypted.astype(np.uint8)

def main():
    """测试程序"""
    try:
        # 读取图像
        img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("图像文件未找到")

        # 创建输出目录
        import os
        os.makedirs("output", exist_ok=True)
        os.makedirs("keys", exist_ok=True)

        # 加密图像
        encryptor = ImageEncryption()
        encrypted = encryptor.encrypt(img)

        # 保存结果到output目录
        cv2.imwrite('output/encrypted.png', encrypted)
        print("🔐 加密完成！文件已保存到 output/encrypted.png")

        # 计算一些评价指标
        original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        encrypted_hist = cv2.calcHist([encrypted], [0], None, [256], [0, 256])
        
        print("📊 原始图像熵:", f"{-np.sum((original_hist/img.size) * np.log2(original_hist/img.size + 1e-10)):.4f}")
        print("📊 加密图像熵:", f"{-np.sum((encrypted_hist/img.size) * np.log2(encrypted_hist/img.size + 1e-10)):.4f}")
        print("🔑 密钥文件已保存到 keys/ 目录")

    except Exception as e:
        print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    main()