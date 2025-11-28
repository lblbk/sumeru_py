import subprocess
import re
from typing import Optional, Literal

class PulseAudioVolumeController:
    """
    统一控制 PulseAudio 音量：支持系统默认输出（sink）或指定应用（sink-input）。
    
    使用方式：
      # 控制系统音量
      sys_vol = PulseAudioVolumeController.for_system()
      
      # 控制特定应用音量
      app_vol = PulseAudioVolumeController.for_app("ALSA plug-in [python3.11]")
    """

    def __init__(
        self,
        target_type: Literal["system", "app"],
        identifier: Optional[str] = None,
        exact_match: bool = True
    ):
        """
        不建议直接调用 __init__，请使用 for_system() 或 for_app()。
        """
        if target_type not in ("system", "app"):
            raise ValueError("target_type 必须是 'system' 或 'app'")
        if target_type == "app" and not identifier:
            raise ValueError("控制应用音量时必须提供 application.name")

        self._target_type = target_type
        self._identifier = identifier          # app name（仅 app 模式使用）
        self._exact_match = exact_match        # 仅 app 模式使用

    @classmethod
    def for_system(cls):
        """创建系统音量控制器"""
        return cls(target_type="system")

    @classmethod
    def for_app(cls, app_name: str, exact: bool = True):
        """创建应用音量控制器"""
        return cls(target_type="app", identifier=app_name, exact_match=exact)

    def _run_pactl(self, *args):
        """安全执行 pactl 命令"""
        try:
            return subprocess.run(
                ["pactl"] + list(args),
                capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError("pactl 命令不可用或失败") from e

    def _get_volume_info(self):
        """
        返回 dict: {"id_or_name": str, "volume": int, "muted": bool}
        """
        if self._target_type == "system":
            sink_name = self._run_pactl("get-default-sink").stdout.strip()

            sinks_output = self._run_pactl("list", "sinks").stdout
            for block in sinks_output.strip().split("\n\n"):
                if f"Name: {sink_name}" in block:
                    vol_match = re.search(r"Volume:.*?(\d+)%", block)
                    volume = int(vol_match.group(1)) if vol_match else None
                    muted = "Mute: yes" in block
                    return {"id_or_name": sink_name, "volume": volume, "muted": muted}
            return None

        else:  # app mode
            inputs_output = self._run_pactl("list", "sink-inputs").stdout
            for block in inputs_output.strip().split("\n\n"):
                if not block.startswith("Sink Input #"):
                    continue

                id_match = re.search(r"Sink Input #(\d+)", block)
                app_match = re.search(r'application\.name = "([^"]*)"', block)
                if not id_match or not app_match:
                    continue

                current_name = app_match.group(1)
                matched = (
                    (self._exact_match and current_name == self._identifier) or
                    (not self._exact_match and self._identifier.lower() in current_name.lower())
                )
                if not matched:
                    continue

                vol_match = re.search(r"/\s*(\d+)%\s*/", block)
                volume = int(vol_match.group(1)) if vol_match else None
                muted = "Mute: yes" in block
                return {"id_or_name": id_match.group(1), "volume": volume, "muted": muted}
            return None

    def get_volume(self) -> Optional[int]:
        info = self._get_volume_info()
        return info["volume"] if info else None

    def set_volume(self, percent: int) -> bool:
        if not (0 <= percent <= 100):
            raise ValueError("音量百分比应在 0–100 之间")
        info = self._get_volume_info()
        if not info:
            return False

        try:
            if self._target_type == "system":
                self._run_pactl("set-sink-volume", "@DEFAULT_SINK@", f"{percent}%")
            else:
                self._run_pactl("set-sink-input-volume", info["id_or_name"], f"{percent}%")
            return True
        except RuntimeError:
            return False

    def mute(self, mute: bool = True) -> bool:
        info = self._get_volume_info()
        if not info:
            return False
        flag = "1" if mute else "0"
        try:
            if self._target_type == "system":
                self._run_pactl("set-sink-mute", "@DEFAULT_SINK@", flag)
            else:
                self._run_pactl("set-sink-input-mute", info["id_or_name"], flag)
            return True
        except RuntimeError:
            return False

    def is_muted(self) -> Optional[bool]:
        info = self._get_volume_info()
        return info["muted"] if info else None

    def is_active(self) -> bool:
        """系统音量始终视为活跃；应用音量需存在 sink-input"""
        if self._target_type == "system":
            return True
        return self._get_volume_info() is not None

    def get_target_info(self) -> Optional[str]:
        """返回目标标识（用于调试）"""
        info = self._get_volume_info()
        if info:
            return f"{self._target_type}: {info['id_or_name']}"
        return None
