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


class ImageDecryption:
    """
    图像解密类
    按加密的逆序实现解密操作
    """
    def __init__(self, block_size=8, base_x=0.12, base_y=0.23, base_z=0.34):
        self.block_size = block_size
        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z
    
    def generate_initial_values(self, encrypted_image):
        """
        使用与加密相同的方式生成初始值
        """
        hash_val = sha384(encrypted_image.tobytes()).digest()
        hash_nums = np.frombuffer(hash_val, dtype=np.uint8)
        mean_val = np.mean(hash_nums)
        
        def xor_sum(nums):
            return reduce(xor, nums)
        
        x0 = self.base_x + ((xor_sum(hash_nums[:8]) + mean_val) % 256) / 256
        y0 = self.base_y + ((xor_sum(hash_nums[8:16]) + mean_val) % 256) / 256
        z0 = self.base_z + ((xor_sum(hash_nums[16:24]) + mean_val) % 256) / 256
        
        return x0 % 1, y0 % 1, z0 % 1

    def inverse_diffuse_image(self, diffused_image, sequence):
        """
        扩散操作的逆操作
        """
        M, N = diffused_image.shape
        S, T = self.block_size, self.block_size
        
        # 分块处理
        blocks = []
        pad_rows = (S - M % S) % S
        pad_cols = (T - N % T) % T
        padded = np.pad(diffused_image, ((0, pad_rows), (0, pad_cols)), mode='constant')
        
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
        
        # 对每个块进行逆扩散操作
        recovered_blocks = [None] * num_blocks
        # 首先将块重排序到扩散时的顺序
        diffused_blocks = [blocks[V[i]] for i in range(num_blocks)]
        
        # 逆序进行异或操作
        for i in range(num_blocks-1, -1, -1):
            Q = diffused_blocks[i]
            if i == 0:
                Q_recovered = np.bitwise_xor(Q, E)
            else:
                Q_recovered = np.bitwise_xor(Q, diffused_blocks[i-1])
            recovered_blocks[V[i]] = Q_recovered
        
        # 重构图像
        rows = (padded.shape[0] // S)
        cols = (padded.shape[1] // T)
        recovered = np.zeros_like(padded)
        
        for idx, block in enumerate(recovered_blocks):
            i = (idx // cols) * S
            j = (idx % cols) * T
            recovered[i:i+S, j:j+T] = block
        
        # 移除padding
        recovered = recovered[:M, :N]
        return recovered

    def inverse_scramble_image(self, scrambled_image, x_seq, y_seq):
        """
        置乱操作的逆操作
        """
        M, N = scrambled_image.shape
        unscrambled = scrambled_image.copy()
        
        # 处理Y序列用于列置乱的逆操作（先进行列操作，因为加密时是后进行的）
        y_indices = np.mod(np.floor(y_seq[:N//2] * 1e13), N).astype(int) + 1
        y_indices = np.unique(y_indices)
        missing_nums = np.setdiff1d(np.arange(1, N+1), y_indices)
        y_indices = np.concatenate([y_indices, np.sort(missing_nums)])
        
        # 列置乱的逆操作（从后往前）
        for j in range(N//2-1, -1, -1):
            if y_indices[j] != y_indices[N-j-1]:
                unscrambled[:, [y_indices[j]-1, y_indices[N-j-1]-1]] = \
                    unscrambled[:, [y_indices[N-j-1]-1, y_indices[j]-1]]
        
        # 处理X序列用于行置乱的逆操作
        x_indices = np.mod(np.floor(x_seq[:M//2] * 1e13), M).astype(int) + 1
        x_indices = np.unique(x_indices)
        missing_nums = np.setdiff1d(np.arange(1, M+1), x_indices)
        x_indices = np.concatenate([x_indices, np.sort(missing_nums)])
        
        # 行置乱的逆操作（从后往前）
        for i in range(M//2-1, -1, -1):
            if x_indices[i] != x_indices[M-i-1]:
                unscrambled[[x_indices[i]-1, x_indices[M-i-1]-1]] = \
                    unscrambled[[x_indices[M-i-1]-1, x_indices[i]-1]]
        
        return unscrambled

    def decrypt(self, encrypted_image):
        """
        解密主函数
        """
        if len(encrypted_image.shape) == 3:
            decrypted_channels = []
            for channel in cv2.split(encrypted_image):
                decrypted_channel = self.decrypt(channel)
                decrypted_channels.append(decrypted_channel)
            return cv2.merge(decrypted_channels)
          # 读取密钥序列
        sequences = np.load('keys/sequences.npz')
        x_seq = sequences['x_seq']
        y_seq = sequences['y_seq']
        z_seq = sequences['z_seq']
        
        # 按加密的逆序进行解密
        # 1. 先进行扩散的逆操作
        undiffused = self.inverse_diffuse_image(encrypted_image, z_seq)
        # 2. 再进行置乱的逆操作
        decrypted = self.inverse_scramble_image(undiffused, x_seq, y_seq)
        
        return decrypted.astype(np.uint8)

def main():
    """测试程序"""
    try:
        # 读取加密图像
        encrypted_img = cv2.imread('output/encrypted.png', cv2.IMREAD_GRAYSCALE)
        if encrypted_img is None:
            raise FileNotFoundError("加密图像文件未找到，请先运行加密程序")

        # 检查密钥文件是否存在
        import os
        if not os.path.exists('keys/sequences.npz'):
            raise FileNotFoundError("密钥文件未找到，请先运行加密程序生成密钥")

        # 创建输出目录
        os.makedirs("output", exist_ok=True)

        # 解密图像
        decryptor = ImageDecryption()
        decrypted = decryptor.decrypt(encrypted_img)

        # 保存结果到output目录
        cv2.imwrite('output/decrypted.png', decrypted)
        print("🔓 解密完成！文件已保存到 output/decrypted.png")

    except Exception as e:
        print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    main()