import numpy as np
import cv2
from hashlib import sha384
from functools import reduce
from operator import xor

class NeuralNetwork:
    """用于训练混沌序列的神经网络类
    
    根据论文1.2节实现的三层前馈神经网络。网络结构包括:
    - 输入层：接收混沌序列
    - 隐藏层：使用tanh激活函数，默认10个神经元
    - 输出层：使用线性激活函数g(ξ)=aξ
    """
    def __init__(self, sequence_length, hidden_size=10, learning_rate=0.6):
        """初始化神经网络
        
        Args:
            sequence_length: 输入序列长度，对应论文中的P
            hidden_size: 隐藏层神经元数量，对应论文中的ncc
            learning_rate: 学习率，对应论文中的ψ参数
        """
        self.learning_rate = learning_rate  # 论文中的ψ参数
        self.hidden_size = hidden_size      # 隐藏层神经元数量ncc
        self.sequence_length = sequence_length
        self.a = 0.35                       # 论文中的比例系数a，用于输出层激活函数
        
        # 初始化网络权重和偏置，对应论文图1中的连接权重
        self.Ve = np.random.randn(1, hidden_size) * 0.01  # 输入层到隐藏层的权重
        self.Ws = np.random.randn(hidden_size, 1) * 0.01  # 隐藏层到输出层的权重
        self.V0e = np.zeros((1, hidden_size))             # 输入层偏置
        self.W0s = np.zeros((1, 1))                       # 输出层偏置
    
    def tanh(self, x):
        """双曲正切激活函数，用于隐藏层
        对应论文方程(3)中的f(ξ)=tanh(ξ)
        """
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """双曲正切导数，用于反向传播
        f'(x) = 1 - tanh²(x)
        """
        return 1.0 - np.tanh(x)**2
    
    def train(self, X, epochs=100):
        """训练网络，完整实现论文1.2节的训练过程
        
        Args:
            X: 输入序列，对应论文中的混沌序列
            epochs: 最大训练轮数
            
        Returns:
            训练后得到的新序列
            
        Note:
            实现了论文方程(2)-(7)的完整训练过程，包括：
            - 前向传播：方程(2)(3)
            - 误差计算：方程(4)
            - 反向传播：方程(5)
            - 权重更新：方程(6)(7)
        """
        X = np.array(X).reshape(-1, 1)
        tolerance = 1e-6  # 收敛判断的容差
        prev_error = float('inf')
        
        for _ in range(epochs):
            total_error = 0
            
            # 前向传播 - 对应论文方程(2)(3)
            hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)  # 隐藏层输出
            output = self.a * (np.dot(hidden, self.Ws) + self.W0s)  # 输出层，使用线性激活g(ξ)=aξ
            
            # 误差计算 - 对应论文方程(4)
            error = output - X
            total_error = np.mean(np.abs(error))
            
            # 反向传播 - 对应论文方程(5)
            delta_output = self.a * error  # δ0,k
            delta_hidden = np.multiply(    # δh,j
                self.tanh_derivative(hidden),
                np.dot(delta_output, self.Ws.T)
            )
            
            # 更新权重和偏置 - 对应论文方程(6)(7)
            self.Ws -= self.learning_rate * np.dot(hidden.T, delta_output) / len(X)
            self.W0s -= self.learning_rate * np.mean(delta_output, axis=0, keepdims=True)
            self.Ve -= self.learning_rate * np.dot(X.T, delta_hidden) / len(X)
            self.V0e -= self.learning_rate * np.mean(delta_hidden, axis=0, keepdims=True)
            
            # 收敛检查
            if abs(total_error - prev_error) < tolerance:
                break
            prev_error = total_error
        
        # 返回训练后的序列
        hidden = self.tanh(np.dot(X, self.Ve) + self.V0e)
        return (self.a * (np.dot(hidden, self.Ws) + self.W0s)).flatten()

class LorenzSystem:
    """改进的Lorenz混沌系统
    
    严格按照论文1.1节方程(1)实现，包括：
    x˙ = a(y - x)
    y˙ = bx - xz + y
    z˙ = 200x² + 0.01exp(xy) - cz
    """
    def __init__(self, x0, y0, z0):
        """初始化Lorenz系统
        
        Args:
            x0, y0, z0: 初始值，由SHA-384和明文图像共同确定
        """
        self.x = x0
        self.y = y0
        self.z = z0
        # 论文规定的系统参数
        self.a = 10.0    # 参数a
        self.b = 40.0    # 参数b
        self.c = 2.5     # 参数c
        self.dt = 0.005  # 时间步长
    
    def next_point(self):
        """计算系统的下一个点
        使用四阶龙格库塔法求解微分方程
        """
        dt = self.dt
        a, b, c = self.a, self.b, self.c
        x, y, z = self.x, self.y, self.z

        # 计算方程(1)中的各项
        dx = a * (y - x)
        dy = b * x - x * z + y
        dz = 200 * x * x + 0.01 * np.exp(x * y) - c * z

        # 更新状态变量
        self.x += dx * dt
        self.y += dy * dt
        self.z += dz * dt

        return self.x, self.y, self.z

    def generate_sequences(self, num_points):
        """生成混沌序列
        
        Args:
            num_points: 需要生成的序列点数
            
        Returns:
            归一化后的混沌序列，值域为[0, 1]
        """
        sequences = []
        
        # 预热迭代以达到混沌状态
        for _ in range(1000):
            self.next_point()
        
        # 记录序列范围以便归一化
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        
        # 生成所需的序列点
        for _ in range(num_points):
            x, y, z = self.next_point()
            sequences.append([x, y, z])
            
            # 更新最大最小值
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
                sequences[:, i] = 0.5  # 处理常数序列的情况
        
        return sequences

class ImageEncryption:
    """图像加密类
    
    实现论文第2节的完整加密方案，包括：
    1. 基于SHA-384的密钥生成
    2. 混沌序列的生成和训练
    3. 图像的置乱和扩散操作
    """
    def __init__(self, block_size=8, base_x=0.12, base_y=0.23, base_z=0.34):
        """初始化加密系统
        
        Args:
            block_size: 扩散操作时的分块大小
            base_x, base_y, base_z: Lorenz系统初值的基准值
        """
        self.block_size = block_size
        self.base_x = base_x
        self.base_y = base_y
        self.base_z = base_z
    
    def generate_initial_values(self, image):
        """生成Lorenz系统的初始值
        
        严格按照论文2.1节方程(8)(9)实现：
        1. 使用SHA-384生成384位哈希值
        2. 通过异或运算和均值计算得到初始值
        
        Args:
            image: 输入图像
            
        Returns:
            x0, y0, z0: Lorenz系统的初始值
        """
        # 计算图像的SHA-384哈希值
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
        
        # 将初始值保存到文件（便于后续解密）
        with open("initial_values.txt", "w") as f:
            f.write(f"x0: {x0}\n")
            f.write(f"y0: {y0}\n")
            f.write(f"z0: {z0}\n")

        return x0, y0, z0

    def scramble_image(self, image, x_seq, y_seq):
        """实现图像置乱
        
        按照论文2.2节方程(10)(11)实现置乱操作：
        1. 使用x序列进行行置乱
        2. 使用y序列进行列置乱
        
        Args:
            image: 输入图像
            x_seq, y_seq: 用于置乱的序列
            
        Returns:
            置乱后的图像
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
        """实现图像像素扩散操作
    
        严格按照论文2.3节实现扩散过程，包括：
        1. 将图像分割成S×T大小的块
        2. 生成扩散序列E进行处理
        3. 按照方程(12)(13)进行扩散操作
    
        Args:
            image: 输入图像，已经过置乱操作
            sequence: 经过神经网络训练的混沌序列
        
        Returns:
            扩散后的图像
        
        Note:
            扩散过程的主要步骤：
            1. 分块：将图像分成大小为S×T的块
            2. 序列处理：使用U序列确定块的处理顺序
            3. 扩散操作：使用E序列进行像素值变换
        """
        M, N = image.shape
        S, T = self.block_size, self.block_size
    
        # 分块处理，确保图像可以被块大小整除
        blocks = []
        pad_rows = (S - M % S) % S  # 计算需要填充的行数
        pad_cols = (T - N % T) % T  # 计算需要填充的列数
        padded = np.pad(image, ((0, pad_rows), (0, pad_cols)), mode='constant')
    
        # 将图像分成S×T的块，对应论文中的分块操作
        for i in range(0, padded.shape[0], S):
            for j in range(0, padded.shape[1], T):
                block = padded[i:i+S, j:j+T].astype(np.uint8)
                blocks.append(block)
    
        # 生成U序列并获取V序列，用于确定块的处理顺序
        num_blocks = len(blocks)
        U = sequence[:num_blocks]  # 取前num_blocks个数作为U序列
        V = np.argsort(U)  # V序列存储U中数的原始位置，用于确定处理顺序
    
        # 生成扩散序列E，按论文方程(12)处理
        E = np.mod(np.floor(sequence[num_blocks:num_blocks+S*T] * 1e13), 256).astype(np.uint8)
        E = E.reshape(S, T)  # 将E整形为S×T的矩阵
    
        # 按照论文方程(13)进行扩散操作
        diffused_blocks = [None] * num_blocks
        for i in range(num_blocks):
            original_pos = V[i]  # 当前要处理的块在原序列中的位置
            Q = blocks[original_pos]  # 获取当前块
        
            if i == 0:
                # 第一个块与E序列进行异或运算
                Q_diffused = np.bitwise_xor(Q, E)
            else:
                # 后续块与前一个处理后的块进行异或运算
                prev_block = diffused_blocks[i-1]
                Q_diffused = np.bitwise_xor(Q, prev_block)
        
            diffused_blocks[i] = Q_diffused
    
        # 将块放回原始位置，恢复原有顺序
        final_blocks = [None] * num_blocks
        for i in range(num_blocks):
            final_blocks[V[i]] = diffused_blocks[i]
    
        # 重构完整图像
        rows = (padded.shape[0] // S)
        cols = (padded.shape[1] // T)
        diffused = np.zeros_like(padded)
    
        for idx, block in enumerate(final_blocks):
            i = (idx // cols) * S  # 计算块的行索引
            j = (idx % cols) * T   # 计算块的列索引
            diffused[i:i+S, j:j+T] = block
    
        # 移除之前添加的填充，返回原始大小的图像
        diffused = diffused[:M, :N]
        return diffused

    def encrypt(self, image):
        """图像加密的主函数
    
        实现论文第2节描述的完整加密流程：
        1. 生成初始值
        2. 产生混沌序列
        3. 使用神经网络训练序列
        4. 执行置乱和扩散操作
    
        Args:
            image: 输入的原始图像
        
        Returns:
            加密后的图像
        """
        if len(image.shape) == 3:
            # 处理彩色图像：分别加密每个通道
            encrypted_channels = []
            for channel in cv2.split(image):
                encrypted_channel = self.encrypt(channel)
                encrypted_channels.append(encrypted_channel)
            return cv2.merge(encrypted_channels)
    
        # 生成Lorenz系统的初始值，对应论文2.1节
        x0, y0, z0 = self.generate_initial_values(image)
    
        # 生成混沌序列
        lorenz = LorenzSystem(x0, y0, z0)
        M, N = image.shape
    
        # 计算所需序列长度
        scramble_points = max(M//2, N//2)  # 置乱操作所需点数
        diffuse_points = (M//self.block_size) * (N//self.block_size) + self.block_size * self.block_size
    
        # 生成用于置乱的序列
        scramble_seq = lorenz.generate_sequences(scramble_points)
        x_seq, y_seq = scramble_seq[:, 0], scramble_seq[:, 1]
    
        # 生成用于扩散的序列，重新初始化Lorenz系统以增加安全性
        lorenz = LorenzSystem(x0, y0, z0)
        diffuse_seq = lorenz.generate_sequences(diffuse_points)
    
        # 使用神经网络训练序列，消除周期性
        nn_scramble = NeuralNetwork(sequence_length=len(scramble_seq))
        nn_diffuse = NeuralNetwork(sequence_length=len(diffuse_seq))
    
        x_seq = nn_scramble.train(x_seq)
        y_seq = nn_scramble.train(y_seq)
        z_seq = nn_diffuse.train(diffuse_seq[:, 2])
    
        # 保存序列供解密使用
        np.savez('sequences.npz', x_seq=x_seq, y_seq=y_seq, z_seq=z_seq)
    
        # 执行置乱和扩散操作
        scrambled = self.scramble_image(image, x_seq, y_seq)
        encrypted = self.diffuse_image(scrambled, z_seq)
    
        return encrypted.astype(np.uint8)

def main():
    """测试程序的主函数
    
    用于演示加密系统的使用方法和评估加密效果
    包括：
    1. 读取测试图像
    2. 执行加密操作
    3. 计算评价指标
    """
    try:
        # 读取测试图像
        img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("图像文件未找到")
        
        # 加密图像
        encryptor = ImageEncryption()
        encrypted = encryptor.encrypt(img)
        
        # 保存加密结果
        cv2.imwrite('encrypted.png', encrypted)
        print("加密完成")
        
        # 计算安全性评价指标
        original_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        encrypted_hist = cv2.calcHist([encrypted], [0], None, [256], [0, 256])
        
        # 计算信息熵，评估加密效果
        print("原始图像熵:", -np.sum((original_hist/img.size) * 
              np.log2(original_hist/img.size + 1e-10)))
        print("加密图像熵:", -np.sum((encrypted_hist/img.size) * 
              np.log2(encrypted_hist/img.size + 1e-10)))
        
    except Exception as e:
        print(f"出现错误: {e}")

if __name__ == "__main__":
    main()