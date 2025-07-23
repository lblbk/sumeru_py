from abc import ABC, abstractmethod
import numpy as np
import h5py

from sumeru_py.core.log import logger

class BaseLoader(ABC):
    @abstractmethod
    def _load_data(self, episode_idx:int):
        pass

    @abstractmethod
    def get_item(self, i):
        """获取指定索引的项"""
        pass

    @abstractmethod
    def get_size(self):
        """获取数据集的大小"""
        pass

class HDF5Loader(BaseLoader):
    def __init__(self, hdf5_path, episode_idx:int = 0, mode="multi"):
        """
        初始化 HDF5Parser 类。
        Args:
            hdf5_path (str): HDF5 文件的路径。
            episode_idx (int): 要解析的 episode 索引，默认为 0。
            mode (str): 模式，默认为 "single"。可以是 "single" 或 "multi"。
        """ 
        self.hdf5_path = hdf5_path
        self.hdf5_data = {}
        self._num_steps = 0
        self.mode = mode
        self._load_data(episode_idx)

    def _format_array(self, arr: np.ndarray) -> str:
        """将 Numpy 数组格式化为单行、固定精度的字符串。"""
        return np.array2string(arr, precision=2, floatmode='fixed', suppress_small=True)

    def _load_from_mode(self):
        """根据模式加载数据"""
        if self.mode == "single":
            return h5py.File(self.hdf5_path, 'r')
        elif self.mode == "multi":
            return h5py.File(self.hdf5_path, 'r')['data']
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _load_data(self, episode_idx):
        """simple parser"""
        try:
            h5_file = h5py.File(self.hdf5_path, 'r')
        except FileNotFoundError:
            print(f"ERROR: Not found '{self.hdf5_path}'")
            return
        
        episode_name = "demo_" + str(episode_idx)
        if  episode_name not in h5_file["data"]:
            print(f"ERROR: Episode '{episode_name}' not found")
            h5_file.close()
            return
        
        try:
            episode_group = h5_file['data'][episode_name]
            action_states = episode_group['actions'][:]
            static_images = episode_group['obs']['image'][:]
            wrist_images = episode_group['obs']['wrist'][:]
            joint_states_with_gripper = episode_group['obs']['joint_states'][:]
            num_steps = static_images.shape[0]
        except KeyError as e:
            print(f"ERROR: {e}")
            h5_file.close()
            return
        
        h5_file.close()

        self._num_steps = num_steps
        self.hdf5_data = {"static_images": static_images,
                          "wrist_images": wrist_images,
                          "action_states": action_states,
                          "joint_states": joint_states_with_gripper[:, :7],
                          "gripper_states": joint_states_with_gripper[:, 7]}
        return 1
    
    def get_item(self, i):
        item = {key[:-1]:self.hdf5_data[key][i] for key in self.hdf5_data.keys()}
        return item

    def get_size(self):
        """获取数据集的大小"""
        return self._num_steps

class RLDSLoader(BaseLoader):
    def __init__(self, ds_name, ds_version:str="1.0.0", split="train", episode_idx:int = 0):
        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
            self.tf = tf
            self.tfds = tfds
        except ImportError:
            raise ImportError(
                "RUN: 'pip install tensorflow' or 'pip install my-library[tf]'"
            )  

        self._load_data(ds_name, ds_version, split, episode_idx)
        # self.rlds_data = 

    def _setup_environment(self):
        """配置 TensorFlow 环境。"""
        if self.config.FORCE_GPU_ALLOW_GROWTH:
            try:
                gpus = self.tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        self.tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth set to True.")
            except RuntimeError as e:
                print(f"Could not set memory growth: {e}")

    def _load_data(self, ds_name, ds_version, split, episode_idx):
        """加载 TFDS 数据集。"""
        try:
            self.ds, ds_info = self.tfds.load(f"{ds_name}:{ds_version}", split=split, with_info=True)
            logger.info(f"ds_info={ds_info}")
        except Exception as e:
            logger.error(f"err = {e}")
            exit()
    
        episode_ds = self.ds.skip(episode_idx).take(1)
        try:
            item_episode = next(iter(episode_ds))
        except StopIteration:
            logger.error("错误：数据集中没有任何数据！")
            return

        steps = list(item_episode['steps'])
        if not steps:
            logger.error("错误：此 Episode 中没有任何步骤。")
            return
            
        self._num_steps = len(steps)
        self.rlds_data = steps

    def get_item(self, i):
        step = self.rlds_data[i]

        observation = step['observation']
        
        # 从 tensor 转换为 numpy 数组
        static_image = observation['image'].numpy()
        static_image = np.rot90(static_image, k=2)
        wrist_image = observation['wrist_image'].numpy()
        joint_state_with_gripper = observation['joint_state'].numpy()

        action_state = step['action'].numpy()

        return {"static_image": static_image,
                "wrist_image": wrist_image,
                "action_state": action_state,
                "joint_state": joint_state_with_gripper[:7],
                "gripper_state": joint_state_with_gripper[7]}
    
    def get_size(self):
        return self._num_steps