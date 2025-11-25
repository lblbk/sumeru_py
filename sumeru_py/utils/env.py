import os
import logging
from typing import Any, Optional, List

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. .env file support disabled.")


class EnvManager:
    """环境变量管理工具类，支持 .env 文件和系统环境变量。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_file: Optional[str] = None):
        if self._initialized:
            return
        self.env_file = env_file or os.getenv("ENV_FILE", ".env")
        self._load_env()
        self._initialized = True

    def _load_env(self):
        """加载 .env 文件（如果存在且 dotenv 可用）"""
        if _DOTENV_AVAILABLE and os.path.isfile(self.env_file):
            load_dotenv(self.env_file, override=True)

    def get(
        self,
        key: str,
        default: Any = None,
        cast: type = str,
        required: bool = False
    ) -> Any:
        """
        获取环境变量。

        :param key: 环境变量名
        :param default: 默认值（若 required=True 则忽略）
        :param cast: 目标类型（str, int, float, bool, list）
        :param required: 是否必填，若为 True 且未设置则抛出 ValueError
        :return: 转换后的值
        """
        value = os.getenv(key)

        if value is None:
            if required:
                raise ValueError(f"Required environment variable '{key}' is not set.")
            return default

        try:
            if cast == bool:
                return self._cast_bool(value)
            elif cast == list:
                return self._cast_list(value)
            elif cast in (int, float):
                return cast(value)
            elif cast == str:
                return value
            else:
                return value  # 保留原始字符串
        except Exception as e:
            raise ValueError(f"Failed to cast environment variable '{key}' to {cast}: {e}")

    @staticmethod
    def _cast_bool(value: str) -> bool:
        """将字符串转换为布尔值（支持常见写法）"""
        return value.lower() in ("true", "1", "yes", "on", "t")

    @staticmethod
    def _cast_list(value: str) -> List[str]:
        """将逗号分隔字符串转为列表，自动 strip 空白"""
        return [item.strip() for item in value.split(",") if item.strip()]

env_mgr = EnvManager()
