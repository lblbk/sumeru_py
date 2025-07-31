import random
import cv2
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

PRESET_COLORS = [
    (230, 25, 75),    # 亮红
    (60, 180, 75),    # 亮绿
    (255, 225, 25),   # 黄色
    (0, 130, 200),    # 蓝色（主推）
    (245, 130, 48),   # 橙色
    (145, 30, 180),   # 紫色
    (70, 240, 240),   # 青色
    (240, 50, 230),   # 粉红
    (210, 245, 60),   # 柠檬黄
    (250, 190, 212),  # 浅粉
    (0, 128, 128),    # 青绿
    (220, 190, 255),  # 浅紫
    (170, 110, 40),   # 棕色
    (255, 250, 200),  # 米白
    (128, 0, 0),      # 深红
    (170, 255, 195),  # 浅绿
    (128, 128, 0),    # 橄榄
    (255, 215, 180),  # 浅橙
    (10, 10, 10),     # 深灰（用于背景类）
    (255, 170, 60),   # 金橙（醒目）
]
COLOR_RATIO = 1
COLOR_POOL = iter(PRESET_COLORS * COLOR_RATIO)
CLASS_NAME_2_COLOR = {}

def get_class_color(class_name):
    global COLOR_POOL, CLASS_NAME_2_COLOR, PRESET_COLORS, COLOR_RATIO
    if class_name not in CLASS_NAME_2_COLOR:
        try:
            color = next(COLOR_POOL)
        except StopIteration:
            # 如果用完，重新开始（理论上不会）
            COLOR_POOL = iter(PRESET_COLORS * COLOR_RATIO)
            color = next(COLOR_POOL)
        CLASS_NAME_2_COLOR[class_name] = color
    return CLASS_NAME_2_COLOR[class_name]

@dataclass
class YOLOConfig:
    backend: str = "ultralytics"
    conf: float = 0.5
    model_path: str = "./model/best.pt"
    labels: List[str] = None
    im_size: Tuple[int, int] = 640, 640


class YOLOInference:
    def __init__(self, config={"backend": "ultralytics",
                               "conf": 0.7,
                               "model_path": "last.pt",
                               "im_size": (720, 1280)}):
        # 将 YAML 数据转换为 dataclass 实例
        self.config = YOLOConfig(**config)

        self.label_name = self.config.labels
        self.im_size = self.config.im_size
        self.model = YOLO(self.config.model_path)

    def infer(self, in_img):
        H, W = in_img.shape[:2]
        image = cv2.resize(in_img, self.im_size)
        
        results = self.model.predict(image, conf=self.config.conf)  # list of 1 Results object
        ret = self.postproc_img(in_img, results, with_mask=True)
        ret_img = self.plot(in_img, ret)
        # ret_img = results[0].plot()
        return ret_img 

    def postproc_img(self, src_img, results: List, with_mask=False):
        result = results[0]
        label_name = self.label_name if self.label_name is not None else list(result.names.values())
        classes: list = result.boxes.cls.cpu().tolist()
        confs: list = result.boxes.conf.cpu().tolist()
        xywh: list = result.boxes.xywh.cpu().tolist() # type: ignore
        
        H, W = src_img.shape[:2]
        scale_x = W / self.im_size[0]
        scale_y = H / self.im_size[1]

        rets = []

        for idx, cls in enumerate(classes):
            conf = confs[idx]
            _x, _y, _w, _h = xywh[idx]

            x = _x * scale_x
            y = _y *  scale_y
            w = _w * scale_x
            h = _h * scale_y

            label_mask = None
            if with_mask:
                label_mask = result.masks.xy[idx]

            if result.names[cls] in label_name:
                label_dict = {
                    'name': result.names[cls],
                    'conf': conf,
                    'center': (int(x), int(y)),
                    'box': (w, h),
                    'xywh': (x, y, w, h),
                    'mask': label_mask,
                    'color': get_class_color(result.names[cls])
                }
                rets.append(label_dict)
        return rets
    
    def plot(self, src_img, rets):
        if len(rets) == 0:
            print("[error] error!!!")
            return src_img

        for ret in rets:
            x, y, w, h = ret["xywh"]
            x1 = int(x - w // 2)
            y1 = int(y - h // 2)
            x2 = int(x + w // 2)
            y2 = int(y + h // 2)

            cv2.rectangle(src_img, (x1, y1), (x2, y2), ret['color'], 2)
            if ret["mask"] is not None:
                overlay = src_img.copy() 
                polygon = [ret["mask"].astype(int)]
                cv2.fillPoly(overlay, polygon, ret['color'])
                alpha = 0.4
                src_img = cv2.addWeighted(overlay, alpha, src_img, 1 - alpha, 0)
        return src_img

