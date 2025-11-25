import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumeru_py.utils.log import logger
from sumeru_py.utils.finder import resource_finder
from sumeru_py.utils.env import env_mgr

def test_logger():
    logger.info("hhhhhh")
    logger.info("qqqqqqqqqq")

def test_resource():
    logger.info(resource_finder.get_project_root())

def test_env():
    print(env_mgr.get("APP_KEY"))
    print(env_mgr.get("APP_KEY", required=True))
    print(env_mgr.get("APP_KEY", default="00000000000"))
    print(env_mgr.get("APP_KEY", default="00000000000", cast=bool, required=True))
    print(env_mgr.get("APP_KE", default="00000000000", required=True))


if __name__ == "__main__":
    # test_logger()
    # test_resource()
    test_env()
