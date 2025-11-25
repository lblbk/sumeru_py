from sumeru_py.robot.dataloader import HDF5Loader, RLDSLoader
from sumeru_py.robot.visualizer import RobotDataVisualizer

def test_visualizer():
    # 1. 创建 HDF5Loader 实例
    hdf5_path = "/home/phibot/Work/robot/dataset/rlds_custom_dataset_builder/dataset/vla_hdf5/place_blue_cube_in_white_plate_demo.hdf5"  # 替换为实际的 HDF5 文件路径
    episode_idx = 0  # 替换为实际的 episode 索引
    
    # dataloader = HDF5Loader(hdf5_path=hdf5_path, episode_idx=episode_idx)
    dataloader = RLDSLoader(ds_name="customer_test", ds_version="3.0.0", episode_idx=episode_idx)
    
    # 2. 创建 HDF5Visualizer 实例
    visualizer = RobotDataVisualizer(dataloader=dataloader)
    
    # 3. 开始实时渲染
    visualizer.render(win_name="Visualization", save_name="1.mp4")

if __name__ == '__main__':
    test_visualizer()
