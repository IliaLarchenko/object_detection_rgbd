from .base_processor import BaseProcessor
import numpy as np
import cv2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from ..utils.pointcloud import create_pointcloud2_msg


class DepthProcessor(BaseProcessor):
    def __init__(
        self,
        camera_intrinsics,
        class_colors,
        publish_pointcloud,
        pointcloud_color_mode="rgb",
    ):
        self.camera_intrinsics = camera_intrinsics
        self.publish_pointcloud = publish_pointcloud
        self.pointcloud_color_mode = pointcloud_color_mode
        self.class_colors = class_colors
        self.active_marker_ids = set()

    def process(self, depth_image, detections_msg, color_image=None):
        K = self.camera_intrinsics["K"]
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        markers = MarkerArray()
        marker_id = 0

        previous_marker_ids = self.active_marker_ids.copy()
        self.active_marker_ids.clear()

        all_points_3d = []
        all_colors = []

        # Process each 2d detection
        for detection in detections_msg.detections:
            bbox = detection.bbox
            class_id = detection.results[0].hypothesis.class_id

            cx_bb = bbox.center.position.x
            cy_bb = bbox.center.position.y
            width_bb = bbox.size_x
            height_bb = bbox.size_y

            xmin = int(np.clip(cx_bb - width_bb / 2, 0, depth_image.shape[1] - 1))
            ymin = int(np.clip(cy_bb - height_bb / 2, 0, depth_image.shape[0] - 1))
            xmax = int(np.clip(cx_bb + width_bb / 2, 0, depth_image.shape[1] - 1))
            ymax = int(np.clip(cy_bb + height_bb / 2, 0, depth_image.shape[0] - 1))

            bbox_area = (xmax - xmin) * (ymax - ymin)
            if bbox_area == 0:
                continue

            # TODO: This is a very rough depth estimation. We should use a more sophisticated method to get the depth of the object.
            # Find the most common depth value in the bounding box
            depth_roi = depth_image[ymin:ymax, xmin:xmax]

            mask = (depth_roi > 0) & np.isfinite(depth_roi)
            if not np.any(mask):
                continue
            valid_depths = depth_roi[mask]
            hist, bin_edges = np.histogram(valid_depths, bins=50)
            max_bin_index = np.argmax(hist)
            depth_mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

            # Assuming everything within 0.3 meters is potentially the object
            depth_threshold = 0.3
            depth_lower = depth_mode - depth_threshold
            depth_upper = depth_mode + depth_threshold
            refined_mask = (depth_roi >= depth_lower) & (depth_roi <= depth_upper)

            # Get the largest connected component and use that as the object mask
            num_labels, labels_im = cv2.connectedComponents(
                refined_mask.astype(np.uint8)
            )

            if num_labels <= 1:
                continue

            label_counts = np.bincount(labels_im.flat)[1:]
            largest_cc = np.argmax(label_counts) + 1
            object_mask = labels_im == largest_cc

            # Generate 3d points from the depth image
            ys, xs = np.nonzero(object_mask)
            zs = depth_roi[ys, xs]

            if zs.size == 0:
                continue

            xs_full = xs + xmin
            ys_full = ys + ymin

            x3d = (xs_full - cx) * zs / fx
            y3d = (ys_full - cy) * zs / fy
            z3d = zs

            points_3d = np.stack((x3d, y3d, z3d), axis=-1)

            min_xyz = points_3d.min(axis=0)
            max_xyz = points_3d.max(axis=0)

            # Create a marker for the 3d bounding box
            marker = Marker()
            marker.header = detections_msg.header
            marker.ns = "object_3d_bounding_boxes"
            marker.id = marker_id
            self.active_marker_ids.add(marker_id)
            marker_id += 1
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01

            # Get color for the class
            color = self.class_colors.get(class_id, [255, 255, 255])
            color_normalized = [c / 255.0 for c in color]
            marker.color.r = color_normalized[0]
            marker.color.g = color_normalized[1]
            marker.color.b = color_normalized[2]
            marker.color.a = 1.0
            marker.lifetime = Duration(sec=0, nanosec=0)

            corners = np.array(
                [
                    [min_xyz[0], min_xyz[1], min_xyz[2]],
                    [max_xyz[0], min_xyz[1], min_xyz[2]],
                    [max_xyz[0], max_xyz[1], min_xyz[2]],
                    [min_xyz[0], max_xyz[1], min_xyz[2]],
                    [min_xyz[0], min_xyz[1], max_xyz[2]],
                    [max_xyz[0], min_xyz[1], max_xyz[2]],
                    [max_xyz[0], max_xyz[1], max_xyz[2]],
                    [min_xyz[0], max_xyz[1], max_xyz[2]],
                ]
            )

            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]

            marker.points = []
            for start, end in edges:
                marker.points.append(
                    Point(x=corners[start][0], y=corners[start][1], z=corners[start][2])
                )
                marker.points.append(
                    Point(x=corners[end][0], y=corners[end][1], z=corners[end][2])
                )

            markers.markers.append(marker)

            if self.publish_pointcloud:
                all_points_3d.append(points_3d)

                if self.pointcloud_color_mode == "rgb" and color_image is not None:
                    colors = color_image[ys_full, xs_full]
                    all_colors.append(colors)
                elif self.pointcloud_color_mode == "class":
                    colors = np.full((len(xs_full), 3), color, dtype=np.float32)
                    all_colors.append(colors)
                else:
                    all_colors.append(None)

        markers_to_delete = previous_marker_ids - self.active_marker_ids
        for marker_id_to_delete in markers_to_delete:
            delete_marker = Marker()
            delete_marker.header = detections_msg.header
            delete_marker.ns = "object_3d_bounding_boxes"
            delete_marker.id = marker_id_to_delete
            delete_marker.action = Marker.DELETE
            markers.markers.append(delete_marker)

        if all_points_3d and self.publish_pointcloud:
            all_points_3d = np.vstack(all_points_3d)
            if self.pointcloud_color_mode in ("rgb", "class") and all_colors:
                all_colors = np.vstack([c for c in all_colors if c is not None]).astype(
                    np.uint8
                )
            else:
                all_colors = None
        else:
            all_points_3d = np.empty((0, 3), dtype=np.float32)
            all_colors = None

        pointcloud_msg = create_pointcloud2_msg(
            header=detections_msg.header, points=all_points_3d, colors=all_colors
        )

        return markers, pointcloud_msg
