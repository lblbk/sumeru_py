import time
import cv2


def is_valid_frame(frame):
    """
    检查帧是否有效（非全黑/全绿）
    """
    if frame is None:
        return False
    
    # 检查全黑帧
    if frame.mean() < 10:  # 平均像素值过低
        return False
    
    # 检查全绿帧 (常见问题)
    if frame.shape[2] == 3:  # 彩色图像
        # 分离通道
        b, g, r = cv2.split(frame)
        # 检查绿色通道是否明显强于其他通道
        if (g.mean() - b.mean() > 50) and (g.mean() - r.mean() > 50):
            return False
    
    # 检查图像方差（模糊/无效图像）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:  # 图像模糊度阈值
        return False
    
    return True

def detect_available_cameras(max_test=10, test_duration=0.5):
    """
    检测所有可用的 USB 摄像头
    
    Params:
        max_test: 最大测试的摄像头索引数量 (默认测试0-9)
        test_duration: 测试每个摄像头的持续时间(秒)

    Returns: 可用摄像头索引列表
    """
    available_cameras = []
    
    print(f"开始检测摄像头 (测试索引 0 到 {max_test-1})...")
    
    for camera_id in range(max_test):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            cap.release()
            continue
            
        print(f"检测到摄像头 {camera_id}, 验证中...")
        
        # 尝试读取几帧以确认摄像头可用
        start_time = time.time()
        frames_read = 0
        valid_frames = 0
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            frames_read += 1
            
            if ret:
                # 检查帧是否有效（非全黑/全绿）
                if is_valid_frame(frame):
                    valid_frames += 1
        
        cap.release()
        
        # 如果有有效帧，则认为摄像头可用
        if valid_frames > 0:
            print(f"  → 摄像头 {camera_id} 可用 (读取 {frames_read} 帧, 有效 {valid_frames} 帧)")
            available_cameras.append(camera_id)
        else:
            print(f"  → 摄像头 {camera_id} 无法获取有效图像")
    
    return available_cameras