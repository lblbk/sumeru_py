import subprocess
import re
from typing import Optional, Dict

import logging

logger = logging.getLogger(__name__)

class PulseAudioAppVolumeController:
    """
    控制指定应用（通过 application.name）的音量。
    支持设置音量、静音、获取当前音量，并自动处理 sink input ID 变化。
    所有数据通过 `pactl list sink-inputs` 解析，兼容 PulseAudio 和 PipeWire。
    """

    def __init__(self, app_name: str, exact: bool = True):
        """
        :param app_name: PulseAudio 中的 application.name，如 "ALSA plug-in [python3.11]"
        :param exact: 是否精确匹配应用名
        """
        self.app_name = app_name
        self.exact = exact

    def _get_sink_input_info(self) -> Optional[Dict[str, object]]:
        """
        解析 `pactl list sink-inputs`，返回匹配应用的 ID 和音量。
        返回 dict: {"id": str, "volume": int, "muted": bool}
        若未找到，返回 None。
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "sink-inputs"],
                capture_output=True,
                text=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("无法调用 pactl 命令，请确认 PulseAudio 或 PipeWire 正在运行") from e

        blocks = result.stdout.strip().split("\n\n")
        for block in blocks:
            if not block.startswith("Sink Input #"):
                continue

            # 提取 ID
            id_match = re.search(r"Sink Input #(\d+)", block)
            if not id_match:
                continue
            sink_id = id_match.group(1)

            # 提取 application.name
            app_name_match = re.search(r'application\.name = "([^"]*)"', block)
            if not app_name_match:
                continue
            current_app_name = app_name_match.group(1)

            # 匹配逻辑
            matched = (
                (self.exact and current_app_name == self.app_name) or
                (not self.exast and self.app_name.lower() in current_app_name.lower())
            )
            if not matched:
                continue

            # 提取音量百分比（取第一个出现的 % 值）
            vol_match = re.search(r"/\s*(\d+)%\s*/", block)
            volume = int(vol_match.group(1)) if vol_match else None

            # 提取 mute 状态
            muted = "Mute: yes" in block

            return {
                "id": sink_id,
                "volume": volume,
                "muted": muted,
                "raw": block
            }

        return None

    def get_volume(self) -> Optional[int]:
        """
        获取当前应用的音量百分比（如 70 表示 70%）。
        如果应用未播放或未找到，返回 None。
        """
        info = self._get_sink_input_info()
        return info["volume"] if info else None

    def is_muted(self) -> Optional[bool]:
        """
        获取当前应用的静音状态。
        返回 True/False，未找到时返回 None。
        """
        info = self._get_sink_input_info()
        return info["muted"] if info else None

    def set_volume(self, percent: int) -> bool:
        """
        设置应用音量（建议 0–100，最大支持 100）。
        """
        if not (0 <= percent <= 100):
            raise ValueError("音量百分比应在 0–100 之间")

        info = self._get_sink_input_info()
        if not info:
            logger.error(f"⚠️ 未找到活跃音频流：application.name = '{self.app_name}'")
            return False

        try:
            subprocess.run(
                ["pactl", "set-sink-input-volume", info["id"], f"{percent}%"],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def mute(self, mute: bool = True) -> bool:
        """
        静音（mute=True）或取消静音（mute=False）。
        """
        info = self._get_sink_input_info()
        if not info:
            return False

        flag = "1" if mute else "0"
        try:
            subprocess.run(
                ["pactl", "set-sink-input-mute", info["id"], flag],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def is_active(self) -> bool:
        """
        判断该应用当前是否有活跃的音频输出。
        """
        return self._get_sink_input_info() is not None

    def get_id(self) -> Optional[str]:
        """
        获取当前 sink input ID（用于调试）。
        """
        info = self._get_sink_input_info()
        return info["id"] if info else None

class PulseAudioSystemVolumeController:
    """
    控制系统默认输出设备（@DEFAULT_SINK@）的音量。
    始终可用（只要音频服务运行），适合作为 fallback。
    """
    @staticmethod
    def _get_default_sink_info():
        """获取默认 sink 的 ID 和音量信息"""
        try:
            sink_name_result = subprocess.run(
                ["pactl", "get-default-sink"],
                capture_output=True, text=True, check=True
            )
            sink_name = sink_name_result.stdout.strip()

            sinks_result = subprocess.run(
                ["pactl", "list", "sinks"],
                capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("pactl 不可用") from e

        blocks = sinks_result.stdout.strip().split("\n\n")
        for block in blocks:
            if f"Name: {sink_name}" in block:
                # 提取音量
                vol_match = re.search(r"Volume:.*?(\d+)%", block)
                volume = int(vol_match.group(1)) if vol_match else None
                muted = "Mute: yes" in block
                return {"name": sink_name, "volume": volume, "muted": muted}
        return None

    def get_volume(self) -> Optional[int]:
        """获取系统音量百分比（0–100+）"""
        info = self._get_default_sink_info()
        return info["volume"] if info else None

    def set_volume(self, percent: int) -> bool:
        """设置系统音量（建议 0–100，最大 150）"""
        if not (0 <= percent <= 150):
            raise ValueError("系统音量建议设为 0–150%")
        try:
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def mute(self, mute: bool = True) -> bool:
        """静音/取消静音系统音量"""
        flag = "1" if mute else "0"
        try:
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", flag], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def is_muted(self) -> Optional[bool]:
        """获取系统静音状态"""
        info = self._get_default_sink_info()
        return info["muted"] if info else None

    def get_sink_name(self) -> Optional[str]:
        """获取默认音频输出设备名称"""
        info = self._get_default_sink_info()
        return info["name"] if info else None
