import time
import cv2
from sumeru_py.cam.cam2dthread import RendererThread, CameraThread

if __name__ == "__main__":
    # 创建摄像头线程 (保留最近5帧作为缓冲)
    camera1 = CameraThread(src=4, buffer_size=5, width=640, height=480).start()
    camera2 = CameraThread(src=10, buffer_size=5, width=640, height=480).start()

    renderer1 = RendererThread([camera1, camera2], window_name="win1").start()

    try:
        while renderer1.is_running():
            frame1 = camera1.get_latest_frame()
            frame2 = camera2.get_latest_frame()
            if all(f is not None for f in [frame1, frame2]):
                print(frame1.shape, frame2.shape)
            time.sleep(0.1)
    finally:
        camera1.stop()
        camera2.stop()
        renderer1.stop()
        cv2.destroyAllWindows()
