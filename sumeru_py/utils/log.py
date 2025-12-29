# my_logger.py

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Type, Any

class DefaultLoggerConfig:
    """
    默认日志配置。用户可以继承此类并覆盖属性以进行自定义。
    """
    LOG_DIR = "logs"
    LOG_FILENAME = "app.log"
    LOG_LEVEL = logging.DEBUG
    CONSOLE_LEVEL = logging.INFO
    BACKUP_INTERVAL = 1
    BACKUP_COUNT = 15

def setup_logger(config_class: Optional[Type[Any]] = None) -> None:
    """
    配置根日志记录器。
    多次调用是安全的：仅当尚未添加 handler 时才配置。
    不使用任何全局变量，状态由 logging 模块自身维护。
    """
    effective_config = config_class or DefaultLoggerConfig

    def get_config_value(attr_name):
        return getattr(effective_config, attr_name, getattr(DefaultLoggerConfig, attr_name))

    log_dir = get_config_value("LOG_DIR")
    log_filename = get_config_value("LOG_FILENAME")
    log_level = get_config_value("LOG_LEVEL")
    console_level = get_config_value("CONSOLE_LEVEL")
    backup_interval = get_config_value("BACKUP_INTERVAL")
    backup_count = get_config_value("BACKUP_COUNT")

    # 使用固定名称作为应用根 logger
    # root_name = "app"
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 只在没有 handlers 时才初始化
    if root_logger.handlers:
        return  # 已配置，直接返回

    formatter = logging.Formatter(
        '%(asctime)s [%(module)s.%(funcName)s:%(lineno)d | %(levelname)s] %(message)s'
    )

    project_root = Path(os.getcwd())
    log_path = project_root / log_dir
    log_path.mkdir(exist_ok=True)

    # 文件处理器
    file_handler = TimedRotatingFileHandler(
        log_path / log_filename,
        when="midnight",
        interval=backup_interval,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    获取一个子 logger
    必须先调用 setup_logger()，否则可能没有 handler（但不会报错）。
    
    示例：
        logger = get_logger(__name__)  # 实际 logger 名为 "app.mymodule"
    """
    return logging.getLogger(name)

# 懒人模式：提供一个默认可用的 logger
setup_logger()
logger = get_logger("default")
