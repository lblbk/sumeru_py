from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['franka_ctl'], # <--- 包名 (对应 src/ 下的目录)
    package_dir={'': 'src'}   # <--- 告诉 distutils 包在 src/ 目录下
)

setup(**setup_args)