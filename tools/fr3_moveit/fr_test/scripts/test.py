#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import cv2
import h5py
import rospy
import moveit_commander
import moveit_msgs.msg
import signal
from franka_ctl.gripper_ctl import FrankaGripperController
from franka_ctl.franka_ctl import MoveGroupPythonInterfaceTutorial



class HDF5Parser:
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path

    def _format_array(self, arr: np.ndarray) -> str:
        """将 Numpy 数组格式化为单行、固定精度的字符串。"""
        return np.array2string(arr, precision=2, floatmode='fixed', suppress_small=True)
    
    def parser_hdf5(self, episode_idx):
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
            static_images = episode_group['obs']['image'][:]
            wrist_images = episode_group['obs']['wrist'][:]
            joint_states_with_gripper = episode_group['obs']['joint_states'][:]
            num_steps = static_images.shape[0]
        except KeyError as e:
            print(f"ERROR: {e}")
            h5_file.close()
            return
        
        h5_file.close()

        joint_states = joint_states_with_gripper[:, :7]
        gripper_states = joint_states_with_gripper[:, 7]

        return {"static_images": static_images,
                "wrist_images": wrist_images,
                "joint_states": joint_states,
                "gripper_states": gripper_states}

def wrap_raw_joint_states(r_states, g_states, g_threshold: float = 0.02):
    ''' pack robot and gripper
    - gripper value: 1 - open, 0 - close
    - g_threshold: status threshold
    '''
    assert len(g_states)==2, "len(g_states)!=2"
    if (g_states[0] > g_threshold) and (g_states[1] > 0.02):
        return r_states + [1]
    else:
        return r_states + [0]

def signal_handler(sig, frame):
    global interrupted
    print('\n[WARNNING] Ctrl+C pressed! Sending stop command...')
    interrupted = True

if __name__ == '__main__':
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    rospy.init_node('franka_gripper_controller_node', anonymous=True)
    tutorial = MoveGroupPythonInterfaceTutorial()
    gripper_controller = FrankaGripperController(gripper_group_name="fr3_hand") 

    # ret = wrap_raw_joint_states(tutorial.get_current_joint_values(), gripper_controller.get_current_joint_values())
    # rospy.loginfo(f"{len(ret)}")

    try:
        START_POSE = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
        TEST_POSE4 = [-0.10904336,0.4151102, -0.05362144,-2.2315097,0.04991567,2.6517367,-0.33372545]
        TEST_POSE5 = [-0.00011038860028906095, -0.7919765398528225, -3.2791931454034974e-05, -2.7587287249221832, -0.0005436418918912347, 1.9663001351250016, 0.7849465617846156]

        hdf5_parser = HDF5Parser("/home/phibot/Work/robot/dataset/rlds_custom_dataset_builder/dataset/vla_hdf5/pickup_blue_box_demo.hdf5")
        eposide_info = hdf5_parser.parser_hdf5(0)

        tutorial.go_to_joint_state(TEST_POSE5, wait=False)
        rospy.sleep(0.5)

        # tutorial.go_to_joint_state(TEST_POSE4, wait=False)
        # rospy.sleep(0.5)

        # tutorial.go_to_joint_state(START_POSE, wait=True)
        # rospy.sleep(0.5)

        # traj_list = eposide_info["joint_states"].tolist()
        # rospy.loginfo(traj_list[0])
        # tutorial.go_to_trajectory(traj_list)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if interrupted:
                rospy.loginfo("Interrupt flag detected. Stopping motion.")
                tutorial.stop() # 在主循环中安全地调用 stop
                # 可以选择在这里 break 或者让循环自然结束 (当 is_shutdown 变为 True)
                break # 立即退出循环
            rate.sleep()

    except Exception as e:
        rospy.logerr("An error occurred: %s", e)
    finally:
        tutorial.stop()
        tutorial.clear_pose_targets()
        rospy.sleep(0.5) # 给停止命令一些时间生效
        moveit_commander.roscpp_shutdown()
        rospy.loginfo(" finally clear finished. ")

