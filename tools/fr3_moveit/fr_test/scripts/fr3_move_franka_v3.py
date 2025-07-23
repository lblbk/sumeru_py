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
from franka_ctl.gripper_ctl import FrankaGripperController

try:
    import franka_gripper.msg
except ImportError:
    rospy.logerr("Cannot import franka_gripper.msg. Make sure franka_ros is installed and sourced.")
    sys.exit(1)

def read_joint_states(file_path):
    """
    从文本文件读取关节状态数据
    每行格式：7个关节角度 + 2个夹爪状态（共9个浮点数）
    """
    joint_states = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            data = line.strip().split(",")[:]
            
            if len(data) != 8:
                rospy.logerr(f"len(data) = {len(data)}")
                continue
                
            try:
                # 将字符串转换为浮点数
                values = list(map(float, data))
                joint_states.append(values)
            except ValueError:
                rospy.logerr("第%d行包含非数字内容", line_num+1)
    
    return joint_states

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

class FrankaGripperClient:
    def __init__(self):
        # Create Action clients
        self.move_client = actionlib.SimpleActionClient('franka_gripper/move', franka_gripper.msg.MoveAction)
        self.grasp_client = actionlib.SimpleActionClient('franka_gripper/grasp', franka_gripper.msg.GraspAction)
        rospy.loginfo("Waiting for franka_gripper action servers...")
        self.move_client.wait_for_server()
        self.grasp_client.wait_for_server()
        rospy.loginfo("Connected to franka_gripper action servers.")

    def move(self, width, speed=0.1):
        """Moves the gripper to a target width."""
        goal = franka_gripper.msg.MoveGoal(width=width, speed=speed)
        rospy.loginfo(f"Sending move goal: width={width}, speed={speed}")
        self.move_client.send_goal(goal)
        if self.move_client.wait_for_result(rospy.Duration(10.0)): # Adjust timeout as needed
            result = self.move_client.get_result()
            if result.success:
                rospy.loginfo("Move action succeeded.")
                return True
            else:
                rospy.logwarn(f"Move action failed: {result.error}")
                return False
        else:
            rospy.logerr("Move action timed out.")
            self.move_client.cancel_goal()
            return False

    def grasp(self, width, force, speed=0.1, epsilon_inner=0.005, epsilon_outer=0.005):
        """Attempts to grasp an object."""
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = epsilon_inner
        goal.epsilon.outer = epsilon_outer

        rospy.loginfo(f"Sending grasp goal: width={width}, force={force}, speed={speed}")
        self.grasp_client.send_goal(goal)
        if self.grasp_client.wait_for_result(rospy.Duration(15.0)): # Grasp might take longer
            result = self.grasp_client.get_result()
            if result.success:
                rospy.loginfo("Grasp action succeeded.")
                return True
            else:
                rospy.logwarn(f"Grasp action failed: {result.error}")
                return False
        else:
            rospy.logerr("Grasp action timed out.")
            self.grasp_client.cancel_goal()
            return False

    def open_gripper(self, speed=0.1):
        """Opens the gripper fully."""
        rospy.loginfo("Opening gripper...")
        return self.move(width=0.08, speed=speed) # Adjust max width if different

    def close_gripper(self, speed=0.1):
        """Closes the gripper fully (moves to width 0)."""
        rospy.loginfo("Closing gripper...")
        return self.move(width=0.0, speed=speed)

class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        # group_name = "panda_arm"
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
        
        move_group = self.move_group
        
         # --- 设置规划时间 ---
        planning_time_limit = 0.05
        move_group.set_planning_time(planning_time_limit)

        last_item = joints_position_lists[0]
        for item in joints_position_lists:
            move_group.go(item, wait=False)
            last_item = item
            move_group.stop()

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

def main():
    try:
        tutorial = MoveGroupPythonInterfaceTutorial()
        # gripper_client = FrankaGripperClient()
        gripper_client = FrankaGripperController(gripper_group_name="fr3_hand")

        cur_pose = tutorial.get_current_pose()
        # print(cur_pose)

        ######################### joint position control #########################
        # START_POSE = [0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397]
        TEST_POSE1 = [8.799028964338746e-05,-0.7853572128158407,9.744812628042526e-05,-2.3562464993713754,3.497643154273149e-05,1.5708436714175793,0.7854168031981299]
        # # TEST_POSE4 = [-0.10904336,0.4151102, -0.05362144,-2.2315097,0.04991567,2.6517367,-0.33372545]
        tutorial.go_to_joint_state(TEST_POSE1, wait=False)

        ######################### fripper control ############################
        # gripper_client.close_gripper()
        # rospy.sleep(0.5)  # 等待机器人稳定
        # gripper_client.open_gripper()
        # rospy.sleep(0.5)
        # gripper_client.grasp(width=0.02, force=10.0, speed=0.1)
        rospy.sleep(0.5)

        ######################### trajectory control #########################
        # file_name = "/home/wibot/Work/robot/franka/catkin_ws/src/test/fr_test/scripts/record.txt"
        # with open(file_name, "r") as file:
        #     trajectory_points_positions = [eval(line.strip()) for line in file]
        # # print(trajectory_points_positions[0][:-1])
        # tutorial.go_to_joint_state(trajectory_points_positions[0][:-1], wait=False)
        # rospy.sleep(0.5)
        # for item in trajectory_points_positions:
        #     tutorial.go_to_joint_state(item[:-1], wait=False)
        #     if item[-1] == 0:
        #         gripper_client.close_gripper()
        #     if item[-1] == 1:
        #         gripper_client.open_gripper()
        #     rospy.sleep(0.05)

        ######################### pose position control #########################
        # ret = tutorial.go_cartesian_path([0.0, 0.04, 0.03])
        # print(f"ret = {ret}")

        # ori = [-0.9239192300457637, 0.3825875210817224, 0.0001888525996874845, 9.69478280388614e-05]
        # xyz = [0.30696541786402304, 0.00017476409436006777, 0.5901366372951665]
        # tutorial.go_to_pose_goal(xyz + ori)

        ######################### gripper control #########################
        # rospy.loginfo("Attempting to open gripper...")
        # if gripper_client.open_gripper():
        #     rospy.sleep(2.0) # Wait a bit

        # rospy.loginfo("Attempting to close gripper...")
        # if gripper_client.close_gripper():
        #     rospy.sleep(2.0)
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

# 定义信号处理函数
def signal_handler(sig, frame):
    global interrupted
    print('\n[WARNNING] Ctrl+C pressed! Sending stop command...')
    interrupted = True
    # 注意：不要在这里直接调用 rospy 或 moveit 的函数，
    # 因为信号处理程序应该尽快返回。设置标志是更安全的方式。

if __name__ == "__main__":
    # 定义一个全局标志来指示是否收到中断信号
    interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    main()
