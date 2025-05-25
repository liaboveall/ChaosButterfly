import numpy as np
import cv2
from hashlib import sha384, sha256, md5
import hmac
import secrets
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import json

class EnhancedSecurityManager:
    """
    增强安全管理器 - 提供密钥管理和完整性验证
    """
    def __init__(self):
        self.salt_length = 32
        self.key_length = 32
        
    def generate_secure_salt(self):
        """生成加密安全的盐值"""
        return secrets.token_bytes(self.salt_length)
    
    def derive_key(self, password, salt, iterations=100000):
        """使用PBKDF2派生密钥"""
        from hashlib import pbkdf2_hmac
        return pbkdf2_hmac('sha256', password.encode(), salt, iterations)
    
    def encrypt_sensitive_data(self, data, password):
        """加密敏感数据（如密钥序列）"""
        salt = self.generate_secure_salt()
        key = self.derive_key(password, salt)
        
        cipher = AES.new(key, AES.MODE_CBC)
        iv = cipher.iv
        
        # 序列化数据
        if isinstance(data, dict):
            data_bytes = json.dumps(data, cls=NumpyEncoder).encode()
        else:
            data_bytes = data
            
        encrypted_data = cipher.encrypt(pad(data_bytes, AES.block_size))
        
        return {
            'salt': salt.hex(),
            'iv': iv.hex(),
            'data': encrypted_data.hex()
        }
    
    def decrypt_sensitive_data(self, encrypted_package, password):
        """解密敏感数据"""
        salt = bytes.fromhex(encrypted_package['salt'])
        iv = bytes.fromhex(encrypted_package['iv'])
        encrypted_data = bytes.fromhex(encrypted_package['data'])
        
        key = self.derive_key(password, salt)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        decrypted_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        
        try:
            return json.loads(decrypted_data.decode())
        except:
            return decrypted_data
    
    def generate_integrity_hash(self, image):
        """生成图像完整性哈希"""
        # 使用多种哈希算法提高安全性
        sha384_hash = sha384(image.tobytes()).hexdigest()
        sha256_hash = sha256(image.tobytes()).hexdigest()
        
        # 组合哈希
        combined = f"{sha384_hash}:{sha256_hash}"
        return sha256(combined.encode()).hexdigest()
    
    def verify_integrity(self, image, expected_hash):
        """验证图像完整性"""
        current_hash = self.generate_integrity_hash(image)
        return hmac.compare_digest(current_hash, expected_hash)

class NumpyEncoder(json.JSONEncoder):
    """NumPy数组JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class RobustImageEncryption:
    """
    鲁棒性增强的图像加密类
    """
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.security_manager = EnhancedSecurityManager()
        
    def multi_round_encryption(self, image, rounds=3):
        """多轮加密增强安全性"""
        encrypted = image.copy()
        round_keys = []
        
        for round_num in range(rounds):
            # 为每轮生成不同的密钥
            round_key = self.generate_round_key(encrypted, round_num)
            round_keys.append(round_key)
            
            # 执行该轮加密
            encrypted = self.single_round_encrypt(encrypted, round_key)
            
        return encrypted, round_keys
    
    def generate_round_key(self, image, round_num):
        """为特定轮次生成密钥"""
        # 结合图像特征和轮次信息
        image_hash = sha384(image.tobytes()).digest()
        round_data = f"round_{round_num}".encode()
        
        combined_hash = sha384(image_hash + round_data).digest()
        hash_nums = np.frombuffer(combined_hash, dtype=np.uint8)
        
        return {
            'x0': 0.1 + (hash_nums[0] % 256) / 256 * 0.8,
            'y0': 0.1 + (hash_nums[1] % 256) / 256 * 0.8,
            'z0': 0.1 + (hash_nums[2] % 256) / 256 * 0.8,
            'round': round_num
        }
    
    def single_round_encrypt(self, image, round_key):
        """单轮加密操作"""
        # 这里可以调用原有的加密逻辑
        # 为了演示，这里使用简化版本
        
        from encrypt import OptimizedImageEncryption, OptimizedLorenzSystem, OptimizedNeuralNetwork
        
        encryptor = OptimizedImageEncryption()
        encryptor.base_x = round_key['x0']
        encryptor.base_y = round_key['y0'] 
        encryptor.base_z = round_key['z0']
        
        return encryptor.encrypt(image, save_analysis=False)
    
    def adaptive_block_size(self, image):
        """根据图像大小自适应调整块大小"""
        M, N = image.shape
        
        # 根据图像大小选择最优块大小
        if M * N < 128 * 128:
            return 4
        elif M * N < 512 * 512:
            return 8
        else:
            return 16

class AdvancedCryptanalysisResistance:
    """
    高级密码分析抗性测试
    """
    @staticmethod
    def differential_analysis(original, encrypted1, encrypted2):
        """差分分析测试"""
        # 计算两个加密图像的差异
        diff = np.abs(encrypted1.astype(np.int16) - encrypted2.astype(np.int16))
        
        # 计算差异统计
        diff_entropy = SecurityAnalyzer.calculate_entropy(diff.astype(np.uint8))
        diff_uniformity = np.std(diff)
        
        return {
            'differential_entropy': diff_entropy,
            'differential_uniformity': diff_uniformity,
            'resistance_score': min(diff_entropy / 8.0, diff_uniformity / 128.0)
        }
    
    @staticmethod
    def avalanche_effect_test(image, encryptor):
        """雪崩效应测试"""
        # 原始图像加密
        encrypted_original = encryptor.encrypt(image, save_analysis=False)
        
        # 修改一个像素
        modified_image = image.copy()
        modified_image[0, 0] = (modified_image[0, 0] + 1) % 256
        encrypted_modified = encryptor.encrypt(modified_image, save_analysis=False)
        
        # 计算变化率
        changed_pixels = np.sum(encrypted_original != encrypted_modified)
        change_rate = changed_pixels / image.size
        
        return {
            'avalanche_rate': change_rate,
            'changed_pixels': changed_pixels,
            'passes_test': change_rate > 0.5  # 理想情况下应该接近50%
        }
    
    @staticmethod
    def frequency_analysis_resistance(encrypted_image):
        """频率分析抗性测试"""
        hist = cv2.calcHist([encrypted_image], [0], None, [256], [0, 256]).flatten()
        
        # 计算卡方统计量
        expected = encrypted_image.size / 256
        chi_square = np.sum((hist - expected)**2 / expected)
        
        # 计算方差
        variance = np.var(hist)
        
        return {
            'chi_square': chi_square,
            'histogram_variance': variance,
            'uniformity_score': 1.0 / (1.0 + variance / expected**2)
        }

class ComprehensiveSecurityTester:
    """
    综合安全性测试器
    """
    def __init__(self):
        self.cryptanalysis = AdvancedCryptanalysisResistance()
        
    def run_comprehensive_test(self, original_image, encrypted_image, encryptor):
        """运行综合安全性测试"""
        print("🔒 开始综合安全性测试...")
        
        results = {}
        
        # 1. 基础统计测试
        results['basic_stats'] = SecurityAnalyzer.full_security_analysis(
            original_image, encrypted_image
        )
        
        # 2. 雪崩效应测试
        print("  ❄️  雪崩效应测试...")
        results['avalanche'] = self.cryptanalysis.avalanche_effect_test(
            original_image, encryptor
        )
        
        # 3. 频率分析抗性
        print("  📊 频率分析抗性测试...")
        results['frequency_resistance'] = self.cryptanalysis.frequency_analysis_resistance(
            encrypted_image
        )
        
        # 4. 差分分析（需要两个稍有不同的图像）
        print("  🔄 差分分析测试...")
        modified_image = original_image.copy()
        modified_image[0, 0] = (modified_image[0, 0] + 1) % 256
        encrypted_modified = encryptor.encrypt(modified_image, save_analysis=False)
        
        results['differential'] = self.cryptanalysis.differential_analysis(
            original_image, encrypted_image, encrypted_modified
        )
        
        # 5. 生成综合评分
        results['overall_score'] = self.calculate_overall_score(results)
        
        return results
    
    def calculate_overall_score(self, results):
        """计算综合安全评分"""
        scores = []
        
        # 信息熵评分 (0-1)
        entropy_score = min(results['basic_stats']['encrypted_entropy'] / 8.0, 1.0)
        scores.append(entropy_score)
        
        # 相关性评分 (0-1, 越低越好)
        correlation_score = max(0, 1.0 - abs(results['basic_stats']['correlation']))
        scores.append(correlation_score)
        
        # 雪崩效应评分 (0-1)
        avalanche_score = min(results['avalanche']['avalanche_rate'] * 2, 1.0)
        scores.append(avalanche_score)
        
        # 频率分析抗性评分 (0-1)
        freq_score = results['frequency_resistance']['uniformity_score']
        scores.append(freq_score)
        
        # 差分分析抗性评分 (0-1)
        diff_score = results['differential']['resistance_score']
        scores.append(diff_score)
        
        overall = np.mean(scores)
        
        return {
            'individual_scores': {
                'entropy': entropy_score,
                'correlation': correlation_score,
                'avalanche': avalanche_score,
                'frequency': freq_score,
                'differential': diff_score
            },
            'overall': overall,
            'grade': self.get_security_grade(overall)
        }
    
    def get_security_grade(self, score):
        """根据评分获取安全等级"""
        if score >= 0.9:
            return "A+ (优秀)"
        elif score >= 0.8:
            return "A (良好)"
        elif score >= 0.7:
            return "B+ (中等偏上)"
        elif score >= 0.6:
            return "B (中等)"
        elif score >= 0.5:
            return "C (及格)"
        else:
            return "D (不及格)"
    
    def generate_security_report(self, results, output_file="keys/comprehensive_security_report.txt"):
        """生成详细的安全性报告"""
        os.makedirs("keys", exist_ok=True)
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write("🦋 ChaosButterfly 综合安全性测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基础统计
            basic = results['basic_stats']
            f.write("📊 基础统计分析:\n")
            f.write(f"  原始图像熵: {basic['original_entropy']:.4f}\n")
            f.write(f"  加密图像熵: {basic['encrypted_entropy']:.4f}\n")
            f.write(f"  相关性: {basic['correlation']:.6f}\n")
            f.write(f"  像素变化率: {basic['pixel_change_rate']:.2f}%\n\n")
            
            # 雪崩效应
            avalanche = results['avalanche']
            f.write("❄️  雪崩效应测试:\n")
            f.write(f"  变化率: {avalanche['avalanche_rate']:.4f}\n")
            f.write(f"  变化像素数: {avalanche['changed_pixels']}\n")
            f.write(f"  测试结果: {'通过' if avalanche['passes_test'] else '未通过'}\n\n")
            
            # 频率分析抗性
            freq = results['frequency_resistance']
            f.write("📈 频率分析抗性:\n")
            f.write(f"  卡方统计量: {freq['chi_square']:.2f}\n")
            f.write(f"  直方图方差: {freq['histogram_variance']:.2f}\n")
            f.write(f"  均匀性评分: {freq['uniformity_score']:.4f}\n\n")
            
            # 差分分析
            diff = results['differential']
            f.write("🔄 差分分析:\n")
            f.write(f"  差分熵: {diff['differential_entropy']:.4f}\n")
            f.write(f"  差分均匀性: {diff['differential_uniformity']:.4f}\n")
            f.write(f"  抗性评分: {diff['resistance_score']:.4f}\n\n")
            
            # 综合评分
            overall = results['overall_score']
            f.write("🎯 综合评分:\n")
            f.write(f"  信息熵评分: {overall['individual_scores']['entropy']:.4f}\n")
            f.write(f"  相关性评分: {overall['individual_scores']['correlation']:.4f}\n")
            f.write(f"  雪崩效应评分: {overall['individual_scores']['avalanche']:.4f}\n")
            f.write(f"  频率抗性评分: {overall['individual_scores']['frequency']:.4f}\n")
            f.write(f"  差分抗性评分: {overall['individual_scores']['differential']:.4f}\n")
            f.write(f"  总评分: {overall['overall']:.4f}\n")
            f.write(f"  安全等级: {overall['grade']}\n\n")
            
            # 建议
            f.write("💡 优化建议:\n")
            if overall['overall'] < 0.8:
                f.write("  - 考虑增加加密轮数\n")
                f.write("  - 优化混沌系统参数\n")
                f.write("  - 增强密钥派生算法\n")
            else:
                f.write("  - 当前加密强度良好\n")
                f.write("  - 建议定期更新密钥\n")
                f.write("  - 考虑添加数字签名\n")

# 导入SecurityAnalyzer（从前面的代码）
from encrypt import SecurityAnalyzer

if __name__ == "__main__":
    print("🔒 高级安全功能测试...")
    
    # 这里可以添加测试代码
    security_manager = EnhancedSecurityManager()
    print("✅ 安全管理器初始化成功")
