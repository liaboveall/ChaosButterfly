import numpy as np
import cv2
import gc
import psutil
import traceback
from typing import Optional, Callable, Any, Generator
from functools import wraps
import warnings
from contextlib import contextmanager
import os
import sys

class MemoryManager:
    """
    内存管理器 - 监控和优化内存使用
    """
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def check_memory_usage(self):
        """检查内存使用情况"""
        current_memory = self.get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if current_memory > self.memory_threshold / (1024 * 1024):
            warnings.warn(f"内存使用量较高: {current_memory:.2f} MB")
            self.force_garbage_collection()
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        collected = gc.collect()
        if collected > 0:
            print(f"🧹 垃圾回收释放了 {collected} 个对象")
    
    def get_memory_report(self) -> dict:
        """获取内存使用报告"""
        current = self.get_memory_usage()
        return {
            'initial_mb': self.initial_memory,
            'current_mb': current,
            'peak_mb': self.peak_memory,
            'increase_mb': current - self.initial_memory
        }

def memory_monitor(func: Callable) -> Callable:
    """内存监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        memory_manager = MemoryManager()
        
        try:
            result = func(*args, **kwargs)
            memory_manager.check_memory_usage()
            return result
        finally:
            report = memory_manager.get_memory_report()
            print(f"📊 {func.__name__} 内存使用: {report['increase_mb']:.2f} MB (+), 峰值: {report['peak_mb']:.2f} MB")
    
    return wrapper

@contextmanager
def error_handler(operation_name: str, fallback_value: Any = None):
    """统一错误处理上下文管理器"""
    try:
        yield
    except FileNotFoundError as e:
        print(f"❌ 文件未找到 ({operation_name}): {e}")
        if fallback_value is not None:
            return fallback_value
        raise
    except PermissionError as e:
        print(f"❌ 权限错误 ({operation_name}): {e}")
        raise
    except MemoryError as e:
        print(f"❌ 内存不足 ({operation_name}): {e}")
        print("💡 建议: 尝试处理更小的图像或关闭其他程序")
        raise
    except ValueError as e:
        print(f"❌ 数值错误 ({operation_name}): {e}")
        raise
    except Exception as e:
        print(f"❌ 未知错误 ({operation_name}): {type(e).__name__}: {e}")
        print("🔍 详细错误信息:")
        traceback.print_exc()
        raise

class ChunkedImageProcessor:
    """
    分块图像处理器 - 处理大型图像时减少内存使用
    """
    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size
        self.memory_manager = MemoryManager()
    
    def process_image_in_chunks(self, image: np.ndarray, 
                               process_func: Callable[[np.ndarray], np.ndarray],
                               overlap: int = 0) -> np.ndarray:
        """
        分块处理图像
        
        Args:
            image: 输入图像
            process_func: 处理函数
            overlap: 块之间的重叠像素数
        """
        h, w = image.shape[:2]
        
        # 如果图像足够小，直接处理
        if max(h, w) <= self.max_chunk_size:
            return process_func(image)
        
        print(f"🧩 图像过大 ({h}x{w})，使用分块处理 (块大小: {self.max_chunk_size})")
        
        # 计算块的数量
        num_chunks_h = (h + self.max_chunk_size - 1) // self.max_chunk_size
        num_chunks_w = (w + self.max_chunk_size - 1) // self.max_chunk_size
        
        # 创建结果数组
        if len(image.shape) == 3:
            result = np.zeros_like(image)
        else:
            result = np.zeros_like(image)
        
        # 处理每个块
        for i in range(num_chunks_h):
            for j in range(num_chunks_w):
                # 计算块的边界
                start_h = i * self.max_chunk_size
                end_h = min((i + 1) * self.max_chunk_size, h)
                start_w = j * self.max_chunk_size
                end_w = min((j + 1) * self.max_chunk_size, w)
                
                # 添加重叠
                chunk_start_h = max(0, start_h - overlap)
                chunk_end_h = min(h, end_h + overlap)
                chunk_start_w = max(0, start_w - overlap)
                chunk_end_w = min(w, end_w + overlap)
                
                # 提取块
                if len(image.shape) == 3:
                    chunk = image[chunk_start_h:chunk_end_h, chunk_start_w:chunk_end_w, :]
                else:
                    chunk = image[chunk_start_h:chunk_end_h, chunk_start_w:chunk_end_w]
                
                # 处理块
                processed_chunk = process_func(chunk)
                
                # 计算在处理后块中的有效区域
                valid_start_h = start_h - chunk_start_h
                valid_end_h = valid_start_h + (end_h - start_h)
                valid_start_w = start_w - chunk_start_w
                valid_end_w = valid_start_w + (end_w - start_w)
                
                # 将处理后的块放回结果
                if len(image.shape) == 3:
                    result[start_h:end_h, start_w:end_w, :] = \
                        processed_chunk[valid_start_h:valid_end_h, valid_start_w:valid_end_w, :]
                else:
                    result[start_h:end_h, start_w:end_w] = \
                        processed_chunk[valid_start_h:valid_end_h, valid_start_w:valid_end_w]
                
                # 清理内存
                del chunk, processed_chunk
                self.memory_manager.check_memory_usage()
                
                print(f"  ✓ 处理完成块 {i+1}/{num_chunks_h}, {j+1}/{num_chunks_w}")
        
        return result

class ProgressTracker:
    """
    进度跟踪器
    """
    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = None
    
    def start(self):
        """开始跟踪"""
        import time
        self.start_time = time.time()
        print(f"🚀 {self.description}")
    
    def update(self, step: int = 1, message: str = ""):
        """更新进度"""
        self.current_step += step
        percentage = (self.current_step / self.total_steps) * 100
        
        # 计算剩余时间
        if self.start_time:
            import time
            elapsed = time.time() - self.start_time
            if self.current_step > 0:
                estimated_total = elapsed * self.total_steps / self.current_step
                remaining = estimated_total - elapsed
                eta_str = f", ETA: {remaining:.1f}s"
            else:
                eta_str = ""
        else:
            eta_str = ""
        
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        print(f"\r  [{progress_bar}] {percentage:.1f}% {message}{eta_str}", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print()  # 换行
    
    def finish(self, message: str = "完成"):
        """完成跟踪"""
        if self.start_time:
            import time
            total_time = time.time() - self.start_time
            print(f"✅ {message} (总耗时: {total_time:.2f}s)")
        else:
            print(f"✅ {message}")

class SafeImageIO:
    """
    安全的图像输入输出类
    """
    @staticmethod
    def safe_imread(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
        """安全读取图像"""
        with error_handler(f"读取图像 {filepath}"):
            if not os.path.exists(filepath):
                print(f"❌ 文件不存在: {filepath}")
                return None
            
            # 检查文件大小
            file_size = os.path.getsize(filepath)
            if file_size > 100 * 1024 * 1024:  # 100MB
                print(f"⚠️  警告: 文件较大 ({file_size / (1024*1024):.1f} MB)")
            
            # 尝试读取
            image = cv2.imread(filepath, flags)
            if image is None:
                print(f"❌ 无法读取图像文件: {filepath}")
                return None
            
            print(f"✅ 成功读取图像: {filepath} ({image.shape})")
            return image
    
    @staticmethod
    def safe_imwrite(filepath: str, image: np.ndarray, 
                    compression_params: Optional[list] = None) -> bool:
        """安全写入图像"""
        with error_handler(f"保存图像 {filepath}"):
            # 创建目录
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 验证图像数据
            if image is None or image.size == 0:
                print(f"❌ 无效的图像数据")
                return False
            
            # 检查数据类型
            if image.dtype != np.uint8:
                print(f"⚠️  转换图像数据类型: {image.dtype} -> uint8")
                image = image.astype(np.uint8)
            
            # 设置压缩参数
            if compression_params is None:
                if filepath.lower().endswith('.png'):
                    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
                elif filepath.lower().endswith('.jpg') or filepath.lower().endswith('.jpeg'):
                    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            
            # 保存图像
            success = cv2.imwrite(filepath, image, compression_params)
            
            if success:
                file_size = os.path.getsize(filepath)
                print(f"✅ 成功保存图像: {filepath} ({file_size / 1024:.1f} KB)")
                return True
            else:
                print(f"❌ 保存图像失败: {filepath}")
                return False

class ValidationTools:
    """
    验证工具类
    """
    @staticmethod
    def validate_image(image: np.ndarray, name: str = "图像") -> bool:
        """验证图像数据"""
        if image is None:
            print(f"❌ {name} 为空")
            return False
        
        if not isinstance(image, np.ndarray):
            print(f"❌ {name} 不是 numpy 数组")
            return False
        
        if image.size == 0:
            print(f"❌ {name} 大小为 0")
            return False
        
        if len(image.shape) not in [2, 3]:
            print(f"❌ {name} 维度错误: {image.shape}")
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            print(f"❌ {name} 通道数错误: {image.shape[2]}")
            return False
        
        # 检查数值范围
        if image.dtype == np.uint8:
            if np.any(image < 0) or np.any(image > 255):
                print(f"⚠️  {name} 像素值超出范围 [0, 255]")
        
        return True
    
    @staticmethod
    def validate_encryption_keys() -> bool:
        """验证加密密钥"""
        required_files = [
            'keys/sequences.npz',
            'keys/initial_values.txt'
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ 缺少密钥文件: {file_path}")
                return False
        
        # 验证序列文件
        try:
            sequences = np.load('keys/sequences.npz')
            required_keys = ['x_seq', 'y_seq', 'z_seq']
            
            for key in required_keys:
                if key not in sequences:
                    print(f"❌ 序列文件缺少键: {key}")
                    return False
                
                seq = sequences[key]
                if len(seq) == 0:
                    print(f"❌ 序列 {key} 为空")
                    return False
            
            print("✅ 密钥验证通过")
            return True
            
        except Exception as e:
            print(f"❌ 密钥验证失败: {e}")
            return False

def create_robust_encrypt_function():
    """
    创建健壮的加密函数
    """
    @memory_monitor
    def robust_encrypt(image_path: str, output_path: str = None, 
                      chunk_size: int = 512, show_progress: bool = True) -> bool:
        """
        健壮的图像加密函数
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径（可选）
            chunk_size: 分块大小
            show_progress: 是否显示进度
        """
        
        # 读取图像
        image = SafeImageIO.safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        
        # 验证图像
        if not ValidationTools.validate_image(image, "输入图像"):
            return False
        
        # 设置输出路径
        if output_path is None:
            output_path = "output/encrypted_robust.png"
        
        # 创建进度跟踪器
        total_steps = 4  # 初始化、序列生成、置乱、扩散
        if show_progress:
            progress = ProgressTracker(total_steps, "健壮加密")
            progress.start()
        
        try:            # 1. 初始化加密器
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
            from encrypt import OptimizedImageEncryption
            encryptor = OptimizedImageEncryption()
            
            if show_progress:
                progress.update(1, "初始化完成")
            
            # 2. 分块处理（如果需要）
            processor = ChunkedImageProcessor(max_chunk_size=chunk_size)
            
            def encrypt_chunk(chunk):
                return encryptor.encrypt(chunk)
            
            if show_progress:
                progress.update(1, "开始加密")
            
            # 3. 执行加密
            encrypted = processor.process_image_in_chunks(image, encrypt_chunk)
            
            if show_progress:
                progress.update(1, "加密完成")
            
            # 4. 验证和保存结果
            if not ValidationTools.validate_image(encrypted, "加密图像"):
                return False
            
            success = SafeImageIO.safe_imwrite(output_path, encrypted)
            
            if show_progress:
                progress.update(1, "保存完成")
                progress.finish("加密成功")
            
            return success
            
        except Exception as e:
            if show_progress:
                progress.finish(f"加密失败: {e}")
            return False
    
    return robust_encrypt

def create_robust_decrypt_function():
    """
    创建健壮的解密函数
    """
    @memory_monitor
    def robust_decrypt(encrypted_path: str, output_path: str = None,
                      validate_keys: bool = True, show_progress: bool = True) -> bool:
        """
        健壮的图像解密函数
        
        Args:
            encrypted_path: 加密图像路径
            output_path: 输出路径（可选）
            validate_keys: 是否验证密钥
            show_progress: 是否显示进度
        """
        
        # 验证密钥
        if validate_keys and not ValidationTools.validate_encryption_keys():
            return False
        
        # 读取加密图像
        encrypted_image = SafeImageIO.safe_imread(encrypted_path, cv2.IMREAD_GRAYSCALE)
        if encrypted_image is None:
            return False
        
        # 验证图像
        if not ValidationTools.validate_image(encrypted_image, "加密图像"):
            return False
        
        # 设置输出路径
        if output_path is None:
            output_path = "output/decrypted_robust.png"        # 创建进度跟踪器
        total_steps = 3  # 初始化、解密、保存
        if show_progress:
            progress = ProgressTracker(total_steps, "健壮解密")
            progress.start()
        
        try:
            # 1. 初始化解密器
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
            from decrypt import OptimizedImageDecryption
            decryptor = OptimizedImageDecryption()
            
            if show_progress:
                progress.update(1, "初始化完成")
            
            # 2. 执行解密
            decrypted = decryptor.decrypt(encrypted_image)
            
            if show_progress:
                progress.update(1, "解密完成")
            
            # 3. 验证和保存结果
            if not ValidationTools.validate_image(decrypted, "解密图像"):
                return False
            
            success = SafeImageIO.safe_imwrite(output_path, decrypted)
            
            if show_progress:
                progress.update(1, "保存完成")
                progress.finish("解密成功")
            
            return success
            
        except Exception as e:
            if show_progress:
                progress.finish(f"解密失败: {e}")
            return False
    
    return robust_decrypt

def main():
    """测试健壮性功能"""
    print("🛡️  ChaosButterfly 健壮性测试")
    print("=" * 50)
    
    # 创建健壮的加密和解密函数
    robust_encrypt = create_robust_encrypt_function()
    robust_decrypt = create_robust_decrypt_function()
    
    # 测试加密
    print("\n🔐 测试健壮加密...")
    encrypt_success = robust_encrypt('lena.png')
    
    if encrypt_success:
        # 测试解密
        print("\n🔓 测试健壮解密...")
        decrypt_success = robust_decrypt('output/encrypted_robust.png')
        
        if decrypt_success:
            print("\n✅ 健壮性测试通过！")
            
            # 比较原图和解密图
            original = SafeImageIO.safe_imread('lena.png', cv2.IMREAD_GRAYSCALE)
            decrypted = SafeImageIO.safe_imread('output/decrypted_robust.png', cv2.IMREAD_GRAYSCALE)
            
            if original is not None and decrypted is not None:
                diff = np.abs(original.astype(np.float32) - decrypted.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                print(f"📊 解密质量:")
                print(f"  • 最大差异: {max_diff}")
                print(f"  • 平均差异: {mean_diff:.3f}")
                print(f"  • 完美匹配: {'是' if max_diff == 0 else '否'}")
        else:
            print("❌ 解密失败")
    else:
        print("❌ 加密失败")

if __name__ == "__main__":
    main()
