from abc import ABC, abstractmethod
import numpy as np
import h5py

class BaseLoader(ABC):
    @abstractmethod
    def __init__(self, hdf5_path, episode_idx:int = 0, mode="multi"):
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
        self._load_hdf5(episode_idx)

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
    
    def _load_hdf5(self, episode_idx):
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
