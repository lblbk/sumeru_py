import os

def check_filename(filename):
    """        
    Ret:
        tuple: (bool, str) 第一个元素表示是否有效，第二个元素是错误消息
    """
    # 检查文件名是否为空
    if not filename:
        return False, "NOT NONE"
    
    # 检查是否包含非法字符
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in illegal_chars:
        if char in filename:
            return False, f"illegal_chars: {char}"
    
    # 检查扩展名（可选）
    valid_extensions = ['.mp4']  # 根据需要修改
    file_ext = os.path.splitext(filename)[1]
    if file_ext not in valid_extensions:
        return False, f"不支持的文件扩展名，支持的扩展名: {', '.join(valid_extensions)}"
    
    return True