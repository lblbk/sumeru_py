import cv2
import threading
from collections import deque
import time
import numpy as np

class CameraThread:
    def __init__(self, src=0, buffer_size=1, width=None, height=None, fps=None):
        """
        初始化摄像头线程
        :param src: 摄像头源 (默认0)
        :param buffer_size: 帧缓冲区大小 (默认只保留最新1帧)
        :param width: 期望宽度 (可选)
        :param height: 期望高度 (可选)
        :param fps: 期望帧率 (可选)
        """
        self.stream = cv2.VideoCapture(src)
        
        # 设置摄像头参数
        if width is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        # 使用双端队列作为环形缓冲区
        self.buffer_size = max(1, buffer_size)  # 至少保留1帧
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()  # 线程锁
        
        # 线程控制
        self.stopped = False
        self.thread = None
        
    def start(self):
        """启动摄像头读取线程"""
        if self.thread is None:
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        return self
    
    def update(self):
        """子线程持续读取帧"""
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            
            # 线程安全地更新缓冲区
            with self.lock:
                self.frame_buffer.append(frame)
    
    def get_latest_frame(self):
        """
        获取最新一帧
        :return: 最新帧 (如果没有帧则返回None)
        """
        with self.lock:
            if len(self.frame_buffer) > 0:
                # 返回一个副本，防止其他线程在渲染时，这个frame被修改
                return self.frame_buffer[-1].copy() 
        return None
    
    def stop(self):
        """停止线程并释放资源"""
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.stream.release()
    
    def is_running(self):
        """检查线程是否在运行"""
        return not self.stopped
    
    def __del__(self):
        """析构函数确保资源释放"""
        self.stop()

class RendererThread:
    def __init__(self, camera_threads, window_name="Renderer"):
        """
        初始化渲染线程
        :param camera_threads: camera_threads 的 CameraThread 实例
        :param window_name: 显示窗口的名称
        """
        if not isinstance(camera_threads, list):
            camera_threads = [camera_threads]
        self.cameras = camera_threads
        self.window_name = window_name
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        """启动渲染线程"""
        self.stopped = False
        self.thread.start()
        return self
    
    def is_running(self):
        """检查线程是否在运行"""
        return not self.stopped

    def update(self):
        """子线程持续获取最新帧并显示"""
        while not self.stopped:
            frames = []
            for cam in self.cameras:
                frame = cam.get_latest_frame()
                if frame is not None:
                    frames.append(frame)
            if all(f is not None for f in frames):
                try:
                    combined_frame = np.hstack(frames)
                    cv2.imshow(self.window_name, combined_frame)
                except Exception as e:
                    print("err=", e)
                    pass
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break

            time.sleep(0.01)

    def stop(self):
        """停止线程并销毁窗口"""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        cv2.destroyAllWindows() # 销毁所有OpenCV窗口
