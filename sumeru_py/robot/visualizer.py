from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import cv2

from sumeru_py.core.log import logger
from sumeru_py.robot.dataloader import BaseLoader

class BaseVisualizer(ABC):
    @dataclass
    class CONFIG:
        FPS: int = 15
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE: float = 0.4
        FONT_COLOR: tuple = (255, 255, 255)  # BGR 格式的颜色 (白色)
        FONT_THICKNESS: int = 1
        TEXT_MARGIN: int = 15
        RESIZE_WINDOW: int = 2

    def __init__(self, config: CONFIG = None, **kwargs):
        config_class = getattr(self.__class__, "CONFIG", BaseVisualizer.CONFIG)
        self._config = config_class(**kwargs) if config is None else config
    
    @property
    def config(self):
        """子类应通过此属性访问配置"""
        return self._config

    @abstractmethod
    def render(self, win_name:str="win1", save:int=0):
        pass

    @abstractmethod
    def save(self, file_name:str="win1"):
        pass


class RobotDataVisualizer(BaseVisualizer):
    '''support hdf5 rlds'''
    def __init__(self, dataloader: BaseLoader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader
    
    def _concatenate_images(self, main_img_rgb: np.ndarray, wrist_img_rgb: np.ndarray) -> np.ndarray:
        # 统一图像高度以便拼接 (将较小的 wrist_image 调整为与 main_image 相同的高度)
        target_height = main_img_rgb.shape[0]
        if wrist_img_rgb.shape[0] != target_height:
            scale = target_height / wrist_img_rgb.shape[0]
            new_width = int(wrist_img_rgb.shape[1] * scale)
            wrist_img_resized = cv2.resize(wrist_img_rgb, (new_width, target_height))
        else:
            wrist_img_resized = wrist_img_rgb

        return np.hstack((main_img_rgb, wrist_img_resized))
    
    def _plot_info(self, img: np.ndarray, joint_state) -> np.ndarray:
        frame_height, frame_width, _ = img.shape
        joint_str = f"{[round(value, 3) for value in joint_state]}"
        (text_width, _), _ = cv2.getTextSize(
            joint_str, self.config.FONT, self.config.FONT_SCALE, self.config.FONT_THICKNESS
        )
        text_x = frame_width - text_width - self.config.TEXT_MARGIN
        text_y = frame_height - self.config.TEXT_MARGIN
        
        cv2.putText(
            img, joint_str, (text_x, text_y),
            self.config.FONT, self.config.FONT_SCALE, self.config.FONT_COLOR,
            self.config.FONT_THICKNESS, cv2.LINE_AA
        )
        return img

    def render(self, win_name:str="win1", save_name:str=None):
        video_writer = None
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        logger.info(f"即将开始实时渲染... 在窗口激活时按 'q' 键退出。")
        
        # --- 循环处理每一帧并显示 ---
        for i in range(self.dataloader.get_size()):
            item = self.dataloader.get_item(i)
            combined_img_rgb = self._concatenate_images(item["static_image"], item['wrist_image'])

            combined_img_rgb = self._plot_info(combined_img_rgb.astype('uint8'), item["joint_state"])

            if self.config.RESIZE_WINDOW is not None:
                combined_img_rgb = cv2.resize(combined_img_rgb, dsize=(combined_img_rgb.shape[1]*self.config.RESIZE_WINDOW, 
                                                                       combined_img_rgb.shape[0]*self.config.RESIZE_WINDOW))

            cv2.resizeWindow(win_name, combined_img_rgb.shape[1], combined_img_rgb.shape[0])

            cv2.imshow(win_name, combined_img_rgb)

            if save_name is not None:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    frame_height, frame_width = combined_img_rgb.shape[:2]
                    video_writer = cv2.VideoWriter(save_name, fourcc, self.config.FPS, (frame_width, frame_height))
                video_writer.write(combined_img_rgb)

            # --- 控制帧率并监听退出键 ---
            delay = int(1000 / self.config.FPS)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                logger.info("用户提前退出。")
                break
        
        cv2.destroyAllWindows()
        if save_name is not None:
            video_writer.release()
        logger.info(f"✅ 实时渲染结束。")

    def save(self, file_name = "win1"):
        return super().save(file_name)
    
