# fr3_moveit

## Installation

1. create ros workspace

```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
```

2. clone rep the `tools/fr3_moveit` to `~/catkin_ws/src`

```
git clone https://github.com/lblbk/sumeru_py
```

3. clone `franka_description` from [franka](https://github.com/frankarobotics/franka_ros) to `~/catkin_ws/src/fr3_moveit`

```
move franka_description ~/catkin_ws/src/fr3_moveit
```

4. install dep

```bash
cd ~/catkin_ws/
rosdep install --from-paths src --ignore-src -y
```

5. 安装程序需要的包

```
ros-noetic-controller-interface
ros-noetic-controller-manager
ros-noetic-moveit-commander
```

比如我是 conda 安装的 ros noetic ，就如下方式安装

```
conda install ros-noetic-controller-manager
```

6. make

```bash
catkin_make
```

## Dir

- `fr_test` 引用控制包代码，脚本程序在 `scripts` 文件夹中, **`src/franka_ctl` moveit 核心控制包**

- `franka_*` 这几个包都是因为需要发送 franka 自定义 `msg` 不得不引入进来

