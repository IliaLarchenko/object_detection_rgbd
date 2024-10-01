#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import PointCloud2
import torch
import numpy as np
from PIL import Image as PILImage

from realsense2_camera_msgs.msg import RGBD
from object_detection_rgbd.models.model_factory import ModelFactory
from object_detection_rgbd.processors.depth_processor import DepthProcessor
from object_detection_rgbd.config.coco_classes_with_colors import (
    COCO_CLASSES,
    CLASS_COLORS,
)
from object_detection_rgbd.utils.visualization import draw_annotations


class ObjectDetection3DNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        self.bridge = CvBridge()

        self.declare_parameters(
            namespace="",
            parameters=[
                # Subscribers
                ("image_topic", "/camera/color/image_raw"),
                ("depth_image_topic", "/camera/aligned_depth_to_color/image_raw"),
                ("camera_info_topic", "/camera/depth/camera_info"),
                ("rgbd_topic", "/camera/rgbd"),
                ("use_rgbd", True),
                # Publishers
                ("detected_objects_topic", "/detected_objects"),
                ("annotated_image_topic", "/annotated_image"),
                ("publish_markers_topic", "/detected_objects_3d"),
                ("output_pointcloud_topic", "/detected_objects_pointcloud"),
                # Detection Model Parameters
                ("model_type", "detr"),
                ("model_name", "facebook/detr-resnet-50"),
                ("use_gpu", True),
                ("confidence_threshold", 0.9),
                ("filter_classes", COCO_CLASSES),
                # Visualization Parameters
                ("publish_annotated_image", True),
                ("publish_3d", True),
                ("publish_pointcloud", True),
                ("pointcloud_color_mode", "rgb"),  # 'rgb', 'class' or 'none'
            ],
        )

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.depth_image_topic = (
            self.get_parameter("depth_image_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.rgbd_topic = (
            self.get_parameter("rgbd_topic").get_parameter_value().string_value
        )
        self.use_rgbd = self.get_parameter("use_rgbd").get_parameter_value().bool_value
        self.detected_objects_topic = (
            self.get_parameter("detected_objects_topic")
            .get_parameter_value()
            .string_value
        )
        self.annotated_image_topic = (
            self.get_parameter("annotated_image_topic")
            .get_parameter_value()
            .string_value
        )
        self.publish_annotated_image = (
            self.get_parameter("publish_annotated_image")
            .get_parameter_value()
            .bool_value
        )
        self.model_name = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )
        self.use_gpu = self.get_parameter("use_gpu").get_parameter_value().bool_value
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        filter_classes_param = (
            self.get_parameter("filter_classes")
            .get_parameter_value()
            .string_array_value
        )

        self.publish_markers_topic = (
            self.get_parameter("publish_markers_topic")
            .get_parameter_value()
            .string_value
        )
        self.output_pointcloud_topic = (
            self.get_parameter("output_pointcloud_topic")
            .get_parameter_value()
            .string_value
        )
        self.publish_3d = (
            self.get_parameter("publish_3d").get_parameter_value().bool_value
        )
        self.pointcloud_color_mode = (
            self.get_parameter("pointcloud_color_mode")
            .get_parameter_value()
            .string_value
        )
        self.publish_pointcloud = (
            self.get_parameter("publish_pointcloud").get_parameter_value().bool_value
        )

        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # Handle Filter Classes
        if filter_classes_param:
            self.filter_classes = [
                cls.decode("utf-8") if isinstance(cls, bytes) else cls
                for cls in filter_classes_param
            ]
            valid_classes = [cls for cls in self.filter_classes if cls in COCO_CLASSES]
            invalid_classes = [
                cls for cls in self.filter_classes if cls not in COCO_CLASSES
            ]

            if invalid_classes:
                for cls in invalid_classes:
                    self.get_logger().warn(
                        f"Class '{cls}' is not part of the COCO dataset and will be ignored."
                    )

            if not valid_classes:
                self.get_logger().error(
                    "No valid classes found in 'filter_classes'. All detections will be ignored."
                )
                self.filter_classes = None
            else:
                self.filter_classes = valid_classes
                self.get_logger().info(
                    f"Filtering detections for classes: {self.filter_classes}"
                )
        else:
            self.filter_classes = None
            self.get_logger().info(
                "No class filtering applied. All detected classes will be processed."
            )

        # Initialize Model
        self.get_logger().info(f"Loading model '{self.model_name}'...")
        model_type = self.get_parameter("model_type").get_parameter_value().string_value
        model_config = {
            "model_type": model_type,
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "filter_classes": self.filter_classes
            if len(self.filter_classes) > 0
            else None,
            "class_names": COCO_CLASSES,
        }
        self.model = ModelFactory.create_model(model_config)
        self.model.load()
        self.get_logger().info("Model loaded successfully.")

        self.class_colors = CLASS_COLORS

        # Initialize Subscribers and Publishers
        if not self.use_rgbd:
            self.image_sub = message_filters.Subscriber(self, Image, self.image_topic)
            self.get_logger().info(f"Subscribed to image topic: {self.image_topic}")

        if self.publish_3d:
            self.camera_info_sub = message_filters.Subscriber(
                self, CameraInfo, self.camera_info_topic
            )
            self.camera_info_sub.registerCallback(self.get_camera_intrinsics)

            self.get_logger().info(
                f"Subscribed to camera info topic: {self.camera_info_topic}"
            )

            if self.use_rgbd:
                self.rgbd_sub = message_filters.Subscriber(self, RGBD, self.rgbd_topic)
                self.rgbd_sub.registerCallback(self.rgbd_callback)
                self.get_logger().info(f"Subscribed to RGBD topic: {self.rgbd_topic}")
            else:
                self.depth_sub = message_filters.Subscriber(
                    self, Image, self.depth_image_topic
                )

                self.get_logger().info(
                    f"Subscribed to depth image topic: {self.depth_image_topic}"
                )

                self.ts = message_filters.ApproximateTimeSynchronizer(
                    [self.image_sub, self.depth_sub], queue_size=3, slop=0.1
                )
                self.ts.registerCallback(self.synchronized_callback)
                self.get_logger().info("Subscribed to image and depth topics.")
        else:
            self.image_sub.registerCallback(self.image_callback)

        self.detected_objects_publisher = self.create_publisher(
            Detection2DArray, self.detected_objects_topic, 5
        )
        self.get_logger().info(
            f"Publishing detections to: {self.detected_objects_topic}"
        )

        if self.publish_annotated_image:
            self.annotated_image_publisher = self.create_publisher(
                Image, self.annotated_image_topic, 5
            )
            self.get_logger().info(
                f"Publishing annotated images to: {self.annotated_image_topic}"
            )

        if self.publish_3d:
            self.marker_pub = self.create_publisher(
                MarkerArray, self.publish_markers_topic, 5
            )
            self.get_logger().info(
                f"Publishing markers to: {self.publish_markers_topic}"
            )
            if self.publish_pointcloud:
                self.pointcloud_pub = self.create_publisher(
                    PointCloud2, self.output_pointcloud_topic, 5
                )
                self.get_logger().info(
                    f"Publishing point clouds to: {self.output_pointcloud_topic}"
                )

        self.camera_intrinsics = None
        self.processor = None

    def get_camera_intrinsics(self, camera_info_msg):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                "K": camera_info_msg.k,
                "width": camera_info_msg.width,
                "height": camera_info_msg.height,
            }
            self.get_logger().info("Camera intrinsics received and stored.")
            self.processor = DepthProcessor(
                camera_intrinsics=self.camera_intrinsics,
                class_colors=self.class_colors,
                publish_pointcloud=self.publish_pointcloud,
                pointcloud_color_mode=self.pointcloud_color_mode,
            )

    def image_callback(self, image_msg):
        return self.synchronized_callback(image_msg, None)

    def rgbd_callback(self, rgbd_msg):
        return self.synchronized_callback(rgbd_msg.rgb, rgbd_msg.depth)

    def synchronized_callback(self, image_msg, depth_msg=None):
        if self.publish_3d and self.camera_intrinsics is None:
            self.get_logger().error("Camera intrinsics not yet received.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed for color image: {e}")
            return

        pil_image = PILImage.fromarray(cv_image)

        outputs = self.model.predict(pil_image)
        detections = self.model.process_outputs(
            outputs, pil_image.size, image_msg.header
        )

        self.detected_objects_publisher.publish(detections)

        if self.publish_annotated_image:
            annotated_image = draw_annotations(cv_image, detections, self.class_colors)
            try:
                annotated_image_msg = self.bridge.cv2_to_imgmsg(
                    annotated_image, encoding="rgb8"
                )
                annotated_image_msg.header = image_msg.header
                self.annotated_image_publisher.publish(annotated_image_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish annotated image: {e}")

        if self.publish_3d and self.processor is not None and depth_msg is not None:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding="passthrough"
                )
                if depth_image.dtype != np.float32:
                    depth_image = depth_image.astype(np.float32) * 0.001
            except Exception as e:
                self.get_logger().error(f"Could not convert depth image: {e}")
                return

            markers, pointcloud_msg = self.processor.process(
                depth_image, detections, cv_image
            )
            if markers is not None and markers.markers:
                self.marker_pub.publish(markers)
            if self.publish_pointcloud and self.pointcloud_pub is not None:
                self.pointcloud_pub.publish(pointcloud_msg)
        elif self.publish_3d:
            self.get_logger().warn("Processor not initialized yet.")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetection3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Object Detection 3D Node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
