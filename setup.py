from setuptools import find_packages, setup
import os
from glob import glob

package_name = "object_detection_rgbd"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ilya Larchenko",
    maintainer_email="ilia.larchenko@gmail.com",
    description="Object detection from RGBD images ROS 2 package",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "object_detection_node = object_detection_rgbd.nodes.object_detection_node:main",
        ],
    },
)
