#!/usr/bin/env python
# Python 2/3 compatibility imports
from __future__ import print_function

import signal
import sys
import copy
import rospy
import actionlib
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from typing import List

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()
        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()

        group_name = "fr3_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

    def go_to_joint_state(self, joints_position: List[float], wait=True):
        if len(joints_position) != 7:
            rospy.logerr(f"len(joints_position) = {len(joints_position)}")
            return False
        
        move_group = self.move_group
        planning_time_limit = 0.05
        move_group.set_planning_time(planning_time_limit)
        move_group.go(joints_position, wait=wait)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joints_position, current_joints, 0.01)
    
    def go_to_trajectory(self, joints_position_lists: List[List[float]]):
        '''使用 move_group.go 模仿透传效果，传进来是一个多个位置
        - joints_position_lists: 需要复现轨迹
        '''
        if (len(joints_position_lists) != 0) and (len(joints_position_lists[0]) != 7):
            rospy.logerr(f"len(joints_position_lists) = {len(joints_position_lists)}")
            rospy.logerr(f"len(joints_position_lists[0]) = {len(joints_position_lists[0])}")
            return False

        rospy.loginfo("Move to start position")
        start_item = joints_position_lists[0]
        self.go_to_joint_state(start_item, wait=False)
        rospy.sleep(0.5)
        
        move_group = self.move_group
        
         # --- 设置规划时间 ---
        planning_time_limit = 0.05
        move_group.set_planning_time(planning_time_limit)

        last_item = joints_position_lists[0]
        for item in joints_position_lists:
            # move_group.go(item, wait=False)
            # last_item = item
            # move_group.stop()
            rospy.loginfo(item)

        current_joints = move_group.get_current_joint_values()
        return all_close(last_item, current_joints, 0.01)

    def get_current_pose(self):
        move_group = self.move_group
        return move_group.get_current_pose()
    
    def get_current_joint_values(self):
        move_group = self.move_group
        return move_group.get_current_joint_values()

    def go_to_pose_goal(self, xyz_quat: List):
        '''移动到指定 [x y z rx ry rz w] 位置，采用四元数表示
        **不推荐使用，除非你知道自己在干什么**
        - xyz_quat: xyz空间坐标+四元数表示转动角度
        '''
        if len(xyz_quat) != 7:
            rospy.logerr(f"len(xyz_quat) = {len(xyz_quat)}") 
            return False
        move_group = self.move_group

        cur_pose = move_group.get_current_pose()

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = xyz_quat[3]
        pose_goal.orientation.y = xyz_quat[4]
        pose_goal.orientation.z = xyz_quat[5]
        pose_goal.orientation.w = xyz_quat[6]
        pose_goal.position.x = xyz_quat[0]
        pose_goal.position.y = xyz_quat[1]
        pose_goal.position.z = xyz_quat[2]

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()

        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose_goal, current_pose, 0.01)

    def go_cartesian_path(self, xyz : List = []):
        '''笛卡尔路径规划, 可以实现沿 x y z 任意坐标轴上移动
        - xyz: 分别在 xyz 轴上移动增量
        '''
        if len(xyz) != 3:
            return False
        move_group = self.move_group

        waypoints = []

        wpose = move_group.get_current_pose().pose
        wpose.position.x += xyz[0]  # Second move forward/backwards in (x)
        wpose.position.y += xyz[1]  # and sideways (y)
        wpose.position.z += xyz[2]  # First move up (z)
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = move_group.compute_cartesian_path(
            waypoints,  # waypoints to follow
            0.01,   # eef_step (1cm resolution)
            avoid_collisions=True   # Enable collision checking
        )

        if fraction < 1.0:
            rospy.logwarn("Could not compute the full cartesian path. Planned fraction: {}".format(fraction))

        if not plan or not plan.joint_trajectory.points:
             rospy.logerr("No trajectory found in the plan!")
             return False
        
        success = move_group.execute(plan, wait=True)

        # 清理: 停止任何残余运动很重要
        move_group.stop()
        move_group.clear_pose_targets()
        return success 

    def display_trajectory(self, plan):
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory)
    
    def stop(self):
        move_group = self.move_group
        move_group.stop()
        # move_group.clear_pose_targets()
        return True
    
    def clear_pose_targets(self):
        move_group = self.move_group
        move_group.clear_pose_targets()