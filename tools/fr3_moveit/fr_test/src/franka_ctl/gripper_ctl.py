#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import rospy
import moveit_commander
import moveit_msgs.msg
# import geometry_msgs.msg # 通常不需要直接用于夹爪关节控制，但MoveIt脚本常导入

class FrankaGripperController:
    def __init__(self, gripper_group_name="hand"): # Franka的夹爪组名通常是 "hand" 或 "panda_hand"
        # 2. 实例化 RobotCommander (获取机器人整体信息)
        self.robot = moveit_commander.RobotCommander()

        # 3. 实例化 PlanningSceneInterface (与规划场景交互)
        self.scene = moveit_commander.PlanningSceneInterface()

        # 4. 实例化 MoveGroupCommander 用于夹爪
        self.gripper_group_name = gripper_group_name
        try:
            self.gripper_group = moveit_commander.MoveGroupCommander(self.gripper_group_name)
        except RuntimeError as e:
            rospy.logerr(f"Error initializing MoveGroupCommander for group '{self.gripper_group_name}': {e}")
            rospy.logerr("Please ensure that the group name is correctly defined in your SRDF and that MoveIt is running.")
            sys.exit(1)


        # rospy.loginfo(f"MoveGroupCommander for gripper '{self.gripper_group_name}' initialized.")
        # rospy.loginfo(f"  Active joints: {self.gripper_group.get_active_joints()}")
        # rospy.loginfo(f"  End effector link: {self.gripper_group.get_end_effector_link()}")
        
        # 允许重规划以增加找到解决方案的几率
        self.gripper_group.allow_replanning(True)
        # 设置规划时间
        self.gripper_group.set_planning_time(5) # 5 秒

    def set_gripper_joint_value(self, width_per_finger):
        """
        Sets the gripper opening by specifying the target value for each finger joint.
        For Franka Hand:
        - 0.0 means closed.
        - 0.04 means fully open (each finger moves 0.04m, total width 0.08m).
        :param width_per_finger: The desired position for each finger joint (0.0 to 0.04).
        """
        if not (0.0 <= width_per_finger <= 0.04):
            rospy.logwarn(f"Requested width per finger {width_per_finger} is out of bounds [0.0, 0.04]. Clamping.")
            width_per_finger = max(0.0, min(width_per_finger, 0.04))

        rospy.loginfo(f"Setting gripper to width per finger: {width_per_finger:.3f} m")
        
        # 获取当前关节值作为基础
        joint_goal = self.gripper_group.get_current_joint_values()
        
        # Franka Hand 通常有两个受控关节: panda_finger_joint1, panda_finger_joint2
        # 它们通常设置为相同的值以实现对称运动
        if len(joint_goal) >= 2: # 确保至少有两个关节
            joint_goal[0] = width_per_finger
            joint_goal[1] = width_per_finger
        elif len(joint_goal) == 1: # 如果组中只定义了一个主动关节
             joint_goal[0] = width_per_finger
        else:
            rospy.logerr("Gripper group has an unexpected number of joints.")
            return False

        self.gripper_group.set_joint_value_target(joint_goal)
        
        # 规划并执行
        # plan_success, plan, planning_time, error_code = self.gripper_group.plan() # plan()返回元组
        # 对于夹爪这种简单运动，可以直接 go()，它内部会规划
        success = self.gripper_group.go(wait=True)

        if success:
            rospy.loginfo("Gripper motion successful.")
        else:
            rospy.logerr("Gripper motion failed.")
        
        self.gripper_group.stop() # 确保没有残余运动
        self.gripper_group.clear_pose_targets() # 执行后清除目标是个好习惯
        rospy.sleep(0.5) # 给状态更新留点时间
        return success

    def open_gripper(self):
        """Opens the gripper fully."""
        rospy.loginfo("Opening gripper fully...")
        return self.set_gripper_joint_value(0.04) # Franka Hand 完全张开时每个指关节为 0.04

    def close_gripper(self, width_per_finger=0.0):
        """
        Closes the gripper.
        :param width_per_finger: Target width per finger. 0.0 for fully closed.
                                  For grasping, a small non-zero value might be better if empty.
        """
        rospy.loginfo(f"Closing gripper to width per finger: {width_per_finger:.3f} m...")
        return self.set_gripper_joint_value(width_per_finger)

    def set_gripper_total_width(self, total_width):
        """
        Sets the gripper to a specified total opening width.
        :param total_width: Desired total distance between finger tips (0.0 to 0.08m).
        """
        if not (0.0 <= total_width <= 0.08):
            rospy.logwarn(f"Requested total width {total_width} is out of bounds [0.0, 0.08]. Clamping.")
            total_width = max(0.0, min(total_width, 0.08))
            
        width_per_finger = total_width / 2.0
        return self.set_gripper_joint_value(width_per_finger)

    def move_gripper_named_target(self, target_name):
        """
        Moves the gripper to a pre-defined named target (e.g., 'open', 'close').
        These must be defined in your robot's SRDF file for the gripper group.
        :param target_name: The name of the target state.
        """
        rospy.loginfo(f"Moving gripper to named target: '{target_name}'")
        
        available_targets = self.gripper_group.get_named_targets()
        if target_name not in available_targets:
            rospy.logerr(f"Named target '{target_name}' not found for gripper group '{self.gripper_group_name}'.")
            rospy.loginfo(f"Available named targets: {available_targets}")
            return False

        self.gripper_group.set_named_target(target_name)
        success = self.gripper_group.go(wait=True)

        if success:
            rospy.loginfo(f"Gripper moved to '{target_name}' successfully.")
        else:
            rospy.logerr(f"Failed to move gripper to '{target_name}'.")

        self.gripper_group.stop()
        self.gripper_group.clear_pose_targets()
        rospy.sleep(0.5)
        return success
    
    def get_current_joint_values(self):
        return self.gripper_group.get_current_joint_values()

    def print_gripper_info(self):
        """Prints information about the gripper group."""
        rospy.loginfo(f"--- Gripper Group: {self.gripper_group_name} ---")
        rospy.loginfo(f"  Active joints: {self.gripper_group.get_active_joints()}")
        rospy.loginfo(f"  Current joint values: {self.gripper_group.get_current_joint_values()}")
        rospy.loginfo(f"  Named targets: {self.gripper_group.get_named_targets()}")
        rospy.loginfo(f"  End effector link: {self.gripper_group.get_end_effector_link()}")
        rospy.loginfo(f"  Planning frame: {self.gripper_group.get_planning_frame()}")
        rospy.loginfo(f"  Pose reference frame: {self.gripper_group.get_pose_reference_frame()}")

if __name__ == '__main__':
    gripper_controller = FrankaGripperController(gripper_group_name="fr3_hand") 
    gripper_controller.print_gripper_info()

    gripper_controller.open_gripper()
    rospy.sleep(1)
    gripper_controller.print_gripper_info() # 打印状态

    rospy.loginfo("\n2. Setting gripper to total width 0.02 m (2 cm)...")
    gripper_controller.set_gripper_total_width(0.02) # 2cm 开口
    rospy.sleep(1)

    gripper_controller.move_gripper_named_target("close")
    rospy.sleep(1)
