import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 1. 定义一个可供用户继承的默认配置类
class DefaultLoggerConfig:
    """
    默认日志配置。用户可以继承此类并覆盖属性以进行自定义。
    """
    LOG_DIR = "logs"
    LOG_FILENAME = "app.log"
    LOG_LEVEL = logging.DEBUG
    CONSOLE_LEVEL = logging.INFO
    MAX_BYTES = 5 * 1024 * 1024  # 5 MB
    BACKUP_COUNT = 3

class AppLogger:
    def __init__(self, config_class=None):
        """
        初始化日志记录器。

        Args:
            config_class (class, optional): 一个包含自定义配置的类。
                                            它应包含 DefaultLoggerConfig 中的部分或全部属性。
                                            如果未提供，则使用 DefaultLoggerConfig。
        """
        # 如果用户没有提供配置类，就使用默认的
        effective_config = config_class or DefaultLoggerConfig

        # 使用 getattr 从配置类中获取值，如果属性不存在，则从默认配置类中获取
        def get_config_value(attr_name):
            return getattr(effective_config, attr_name, getattr(DefaultLoggerConfig, attr_name))

        log_dir = get_config_value("LOG_DIR")
        log_filename = get_config_value("LOG_FILENAME")
        log_level = get_config_value("LOG_LEVEL")
        console_level = get_config_value("CONSOLE_LEVEL")
        max_bytes = get_config_value("MAX_BYTES")
        backup_count = get_config_value("BACKUP_COUNT")

        self.logger = logging.getLogger(f"app_logger_{id(effective_config)}")
        self.logger.setLevel(log_level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] | %(message)s'
        )

        project_root = Path(os.getcwd())
        log_path = project_root / log_dir
        log_path.mkdir(exist_ok=True)

        # 文件处理器
        file_handler = RotatingFileHandler(
            log_path / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def __getattr__(self, name):
        if hasattr(self.logger, name):
            return getattr(self.logger, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

logger = AppLogger()
