# object_detection_rgbd

A ROS2 package for 2D and 3D object detection using RGB-D camera data.

## Overview

`object_detection_rgbd` is a ROS2 package that performs real-time 2D and 3D object detection using data from depth cameras. It leverages pre-trained models to detect objects and can convert 2D detections into 3D bounding boxes and point clouds. The package supports any COCO detection model with minimal adjustments and allows filtering of specific classes.

## Features

- **2D Object Detection**: Performs detection using only color images.
- **3D Object Detection**: Converts 2D detections into 3D bounding boxes and point clouds using depth data.
- **RGB-D Support**: Subscribes directly to RGB-D topics if available.
- **Flexible Model Integration**: Supports any COCO detection model with minimal porting.
- **Class Filtering**: Option to detect all classes or a specified subset.
- **Visualization**:
  - Annotated images with bounding boxes.
  - 3D markers for visualization in RViz.
  - Point clouds of detected objects.

## Demo

Check out a demo video of the package in action:

[![Object Detection](https://img.shields.io/badge/Object%20Detection%20RGBD-black?logo=x)](https://x.com/IliaLarchenko/status/1840576329466671114)

## Installation

### Prerequisites

- **ROS2 Jazzy**: Ensure you have ROS2 Jazzy installed. [Installation Guide](https://docs.ros.org/en/jazzy/Installation.html)
- **Depth Camera**: Tested with Intel RealSense D435 camera.

### Steps

1. **Clone the Repository**

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/IliaLarchenko/object_detection_rgbd.git
   ```

2. **Source ROS2**

   ```bash
   source /opt/ros/jazzy/setup.bash
   ```

3. **Install Dependencies**

   ```bash
   rosdep install -i --from-path src --rosdistro jazzy -y
   ```

4. **Build the Workspace**

   ```bash
   colcon build --symlink-install
   ```

5. **Source the Workspace**

   ```bash
   source install/setup.bash
   ```

6. **Create and Activate a Virtual Environment**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

7. **Install Python Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Source ROS2 and the Workspace**

   ```bash
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   source .venv/bin/activate
   ```

2. **Launch the Object Detection Node**

   ```bash
   ros2 launch object_detection_rgbd object_detection.launch.py
   ```

   *Modify topics or parameters in the launch file if necessary.*

3. **It works now, you can visualize it in RViz**

## Configuration

Adjust parameters in the `object_detection.launch.py` file to suit your setup:

- **Topics**
  - `image_topic`: Color image topic (default: `/camera/color/image_raw`).
  - `depth_image_topic`: Depth image topic (default: `/camera/aligned_depth_to_color/image_raw`).
  - `camera_info_topic`: Camera info topic (default: `/camera/depth/camera_info`).
  - `rgbd_topic`: RGB-D topic (if using RGB-D input).
  - `use_rgbd`: Set to `True` to subscribe to RGB-D topic instead of separate image and depth topics.

- **Publishers**
  - `detected_objects_topic`: Topic to publish detected objects.
  - `annotated_image_topic`: Topic for annotated images.
  - `publish_markers_topic`: Topic for 3D markers.
  - `output_pointcloud_topic`: Topic for point clouds of detected objects.

- **Detection Model Parameters**
  - `model_type`: Type of detection model (e.g., `detr`).
  - `model_name`: Name of the pre-trained model (e.g., `facebook/detr-resnet-50`).
  - `use_gpu`: Enable GPU acceleration (`True` or `False`).
  - `confidence_threshold`: Minimum confidence for detections (e.g., `0.9`).
  - `filter_classes`: List of classes to detect (e.g., `['person']`).

- **Visualization Parameters**
  - `publish_annotated_image`: Publish images with bounding boxes (`True` or `False`).
  - `publish_3d`: Publish 3D markers (`True` or `False`).
  - `publish_pointcloud`: Publish point clouds (`True` or `False`).
  - `pointcloud_color_mode`: Color mode for point clouds (`'rgb'`, `'class'`, or `'none'`).

## Usage Tips

- **Adjusting Topics**: Ensure the topics match those provided by your camera. Modify them in the launch file if they differ.
- **Model Selection**: You can switch to different COCO models by changing the `model_name` parameter.
- **Filtering Classes**: To detect specific classes, set the `filter_classes` parameter with the desired class names.
- **Visualization**: Use RViz to visualize 3D markers and point clouds. Ensure the relevant parameters are set to `True`.

## Disclaimer

- **Hobby Project**: This package is developed as a hobby and is not professionally maintained.
- **Performance**: Performance optimization has not been a focus, I am pretty sure you can do it better.
- **Testing**: Tested only with Intel RealSense D435 camera on ROS2 Jazzy.


## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
