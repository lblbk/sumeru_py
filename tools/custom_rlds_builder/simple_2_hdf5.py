"""
此脚本用于生成 openvla 数据集

这里的输入是文件夹路径，文件夹名是当前任务名称，文件夹中是每个 hdf5 文件是一条数据（这里和 libero 稍有不同）

"""
import os
import h5py
import numpy as np
import argparse

from PIL import Image
# from LIBERO_Spatial.conversion_utils import MultiThreadedDatasetBuilder

class CONFIG:
    img_h = 256
    img_w = 256


def parse_path(dir_path: str, ds_path: str, key_map: dict):
    '''convert hdf5 to vla neede hdf5 format 

    Args:
        key_map: map for hdf5 

    '''
    paths = os.listdir(dir_path)
    if paths[0] == "":
        print("[ERROR] path none")
        return 
    
    dir_name = os.path.basename(dir_path).split('/')[-1]
    if not os.path.exists(ds_path):
        os.makedirs(ds_path)
    f_hdf5 = h5py.File(f"{ds_path}/{dir_name}_demo.hdf5", "w")

    data_group = f_hdf5.create_group("data")

    print("======== 开始处理 ========")
    for i, sample in enumerate(paths):
        item_name = f"demo_{i}"
        demo_group = data_group.create_group(item_name)
        with h5py.File(os.path.join(dir_path, sample), "r") as F:
            print(f"[processing] {os.path.join(dir_path, sample)}")
            # action_diff = F["action"][()]   # action 改为正常值 去除第一个 最后一个值设为 agent_pos[-1] + action[-1]
            action_diff = F[key_map["action_diff"]][()]   # action 改为正常值 去除第一个 最后一个值设为 agent_pos[-1] + action[-1]
            states = F[key_map["states"]][()]
            static_images = F[key_map["static_images"]][()]
            gripper_images = F[key_map["gripper_images"]][()]
            gripper = F[key_map["gripper"]][()]

        re_static_images = np.zeros((len(static_images), CONFIG.img_h, CONFIG.img_w, 3))
        for i, item in enumerate(static_images):
            img = Image.fromarray(item)
            re_img = img.resize((CONFIG.img_h, CONFIG.img_w))
            re_static_images[i] = np.array(re_img, dtype=np.uint8)
        
        re_gripper_images = np.zeros((len(gripper_images), CONFIG.img_h, CONFIG.img_w, 3))
        for i, item in enumerate(gripper_images):
            img = Image.fromarray(item)
            re_img = img.resize((CONFIG.img_h, CONFIG.img_w))
            re_gripper_images[i] = np.array(re_img, dtype=np.uint8)

        gripper_2_dim = gripper.reshape(gripper.shape[0], 1)
        full_states = np.concatenate((states, gripper_2_dim), axis=1)

        actions = action_diff
        actions[:-1] = states[1:]
        # actions[-1] = states[-1] + action_diff[-1]
        actions[-1] = states[-1]

        action_gripper = np.zeros_like(gripper)
        action_gripper[:-1] = gripper[:-1]
        action_gripper[-1] = gripper[-1]

        action_gripper_2_dim = action_gripper.reshape(action_gripper.shape[0], 1)
        full_actions = np.concatenate((actions, action_gripper_2_dim), axis=1)

        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("image", data=re_static_images)
        obs_group.create_dataset("wrist", data=re_gripper_images)
        obs_group.create_dataset("joint_states", data=full_states)

        demo_group.create_dataset("actions", data=full_actions)

    f_hdf5.close()
    print("======== 处理完成 ========")

def filter_hdf5(dir_path):
    paths = os.listdir(dir_path)
    if paths[0] == "":
        print("[ERROR] path none")
        return 
    
    # dir_name = os.path.basename(dir_path).split('/')[-1]
    # f_hdf5 = h5py.File(f"{ds_path}/{dir_name}_demo.hdf5", "w")

    # data_group = f_hdf5.create_group("data")
    idx= 0
    for i, sample in enumerate(paths):
        # item_name = f"demo_{i}"
        # demo_group = data_group.create_group(item_name)
        try:
            with h5py.File(os.path.join(dir_path, sample), "r") as F:    
                print(sample)
                idx += 1
                # pass
        except Exception as e:
            print(f"e = {e}")
            print(os.path.join(dir_path, sample))
    print("===============", idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始 hdf5 文件夹转为 openvla hdf5 格式")
    parser.add_argument("--file_path", "-f", type=str, help="输入文件夹路径", default="/home/phibot/Work/robot/dataset/autopick_git/tmp/place_blue_cube_in_white_plate")
    # parser.add_argument("--file_path", "-f", type=str, help="输入文件夹路径", default="/home/phibot/Work/robot/dataset/rlds_custom_dataset_builder/dataset/put_box_to_balabala")
    parser.add_argument("--save_path", "-s", type=str, help="保存数据集路径", default="/home/phibot/Work/robot/dataset/rlds_custom_dataset_builder/dataset/vla_hdf5")
    parser.add_argument("--filter_hdf5", type=int, help="筛选数据集", default=0)

    args = parser.parse_args()

    if args.filter_hdf5:
        filter_hdf5(args.file_path)

    try:
        hdf5_map_dict = {
            "action_diff": "action_joint",
            "states": "joint",
            "static_images": "D435I_color",
            "gripper_images": "D435_color",
            "gripper": "gripper"
        }

        parse_path(args.file_path, args.save_path, hdf5_map_dict)
    except Exception as e:
        print("e = ", e)
