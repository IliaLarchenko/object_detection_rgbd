from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Declare launch arguments if needed
    use_sim_time = LaunchConfiguration("use_sim_time", default="false")

    # Image Transport Republish Node: Compressed -> Raw
    color_decompress = Node(
        package="image_transport",
        executable="republish",
        name="color_decompress",
        arguments=[
            "compressed",
            "raw",
            "--ros-args",
            "--remap",
            "in:=/camera/color/image_raw/compressed",
            "--remap",
            "out:=/camera/color/image_raw_decompressed",
        ],
        output="screen",
    )

    # Image Transport Republish Node for Depth: Compressed -> Raw (Optional)
    depth_decompress = Node(
        package="image_transport",
        executable="republish",
        name="depth_decompress",
        arguments=[
            "compressed",
            "raw",
            "--ros-args",
            "--remap",
            "in:=/camera/aligned_depth_to_color/image_raw/compressed",
            "--remap",
            "out:=/camera/aligned_depth_to_color/image_raw_decompressed",
        ],
        output="screen",
    )

    object_detection_node = Node(
        package="object_detection_rgbd",
        executable="object_detection_node",
        name="object_detection_node",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "image_topic": "/camera/color/image_raw_decompressed",
                "depth_image_topic": "/camera/aligned_depth_to_color/image_raw_decompressed",
                "camera_info_topic": "/camera/depth/camera_info",
                "rgbd_topic": "/camera/rgbd",
                "use_rgbd": False,  # Whether to subscribe to rgbd topic instead of image and depth topics
                # Publishers
                "detected_objects_topic": "/detected_objects",
                "annotated_image_topic": "/annotated_image",
                "publish_markers_topic": "/detected_objects_3d",
                "output_pointcloud_topic": "/detected_objects_pointcloud",
                # Detection Model Parameters
                "model_type": "detr",
                "model_name": "facebook/detr-resnet-50",
                "use_gpu": True,
                "confidence_threshold": 0.9,
                # 'filter_classes': ['person'],
                # Visualization Parameters
                "publish_annotated_image": True,
                "publish_3d": True,
                "publish_pointcloud": True,
                "pointcloud_color_mode": "rgb",  # 'rgb', 'class' or 'none'
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [color_decompress, depth_decompress, object_detection_node]
    )
