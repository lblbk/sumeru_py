#!/usr/bin/env python
# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import zmq
import json
import rospy
import moveit_commander
import rospy
import moveit_commander

from franka_ctl.gripper_ctl import FrankaGripperController
from franka_ctl.franka_ctl import MoveGroupPythonInterfaceTutorial

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

class JointServer:
    def __init__(self, timeout=5):
        """初始化 ZMQ 上下文和套接字"""
        print("[Server] 初始化...")
        self.context = zmq.Context()
        self.timeout = timeout * 1000   # millisecond

        # REP 套接字：接收请求，发送状态
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(REQ_REP_ADDRESS)
        # print(f"[Server] REP Socket 绑定在 {REQ_REP_ADDRESS}")

        # PULL 套接字：接收处理后的结果
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(PUSH_PULL_ADDRESS)
        self.pull_socket.RCVTIMEO = self.timeout  # 设置接收超时
        # print(f"[Server] PULL Socket 绑定在 {PUSH_PULL_ADDRESS}")

    def _get_current_joint_state(self):
        joint_position = wrap_raw_joint_states(tutorial.get_current_joint_values(), gripper.get_current_joint_values())
        return joint_position

    def _process_received_result(self, result):
        """处理从客户端接收到的结果 - 需要替换为实际逻辑"""
        print(f"[Server] 收到处理后的结果: {result}")
        # tutorial.go_to_joint_state(result, wait=False)
        return 1

    def run_forever(self):
        """持续运行，处理请求和结果"""
        print("[Server] 开始监听请求...")
        try:
            while True:
                # 1. 处理获取关节状态的请求
                print("\n[Server] wait for client request...")
                request_bytes = self.rep_socket.recv()
                request_str = json.loads(request_bytes.decode('utf-8'))
                print(f"[Server] receive: {request_str}")

                if "get_joint_state" in request_str:
                    current_state = self._get_current_joint_state()
                    print("current_state", current_state)
                    state_bytes = json.dumps(current_state).encode('utf-8')
                    self.rep_socket.send(state_bytes)
                    continue
                
                if "set_joint_state" in request_str:
                    ret = self._process_received_result(request_str["set_joint_state"])
                    state_bytes = json.dumps(ret).encode('utf-8')
                    self.rep_socket.send(state_bytes)
                    continue

                print(f"[Server] Unknown request: {request_str}")
                # self.rep_socket.send(b"Unknown request")

                # # 2. 等待并处理来自客户端的结果
                # print("[Server] 等待来自 Client 的处理结果...")
                # try:
                #     result_bytes = self.pull_socket.recv()
                #     result_data = json.loads(result_bytes.decode('utf-8'))
                #     self._process_received_result(result_data)
                # except zmq.error.Again:
                #     print(f"[Server] 警告: 等待结果超时 ({self.timeout/1000}秒)，进入下一个循环")

        except KeyboardInterrupt:
            print("\n[Server] 检测到中断信号...")
        except Exception as e:
            print(f"[Server] 发生错误: {e}")
        finally:
            self.close()

    def close(self):
        """关闭套接字和上下文"""
        print("[Server] 关闭套接字和上下文...")
        self.rep_socket.close()
        self.pull_socket.close()
        self.context.term()
        print("[Server] 服务已关闭。")

# config.py
REQ_REP_PORT = 5555
PUSH_PULL_PORT = 5556
# SERVER_ADDRESS = "localhost" # 或者服务器的 IP 地址

REQ_REP_ADDRESS = f"tcp://*:{REQ_REP_PORT}" # 服务器端通常用 * 监听所有接口
PUSH_PULL_ADDRESS = f"tcp://*:{PUSH_PULL_PORT}"

# --- 主程序 ---
if __name__ == "__main__":
    rospy.init_node('infer_node')
    try:
        gripper = FrankaGripperController(gripper_group_name="fr3_hand") 
        tutorial = MoveGroupPythonInterfaceTutorial()

        # gripper.open_gripper()
        # rospy.sleep(0.5)

        server = JointServer()
        server.run_forever()
    except Exception as e:
        rospy.logerr("An error occurred: %s", e)
    finally:
        tutorial.stop()
        tutorial.clear_pose_targets()
        rospy.sleep(0.5) # 给停止命令一些时间生效
        moveit_commander.roscpp_shutdown()
        rospy.loginfo(" finally clear finished. ")
