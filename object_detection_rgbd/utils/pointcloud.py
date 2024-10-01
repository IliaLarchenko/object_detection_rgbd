from sensor_msgs.msg import PointCloud2, PointField
import numpy as np


def create_pointcloud2_msg(header, points, colors=None):
    """
    Creates a PointCloud2 message from a set of points and optional colors.

    Args:
        header (std_msgs.msg.Header): The header for the point cloud.
        points (numpy.ndarray): A 3xN numpy array of points in the format [x, y, z].
        colors (numpy.ndarray, optional): A 3xN numpy array of colors in the format [r, g, b].

    Returns:
        sensor_msgs.msg.PointCloud2: A PointCloud2 message.
    """
    if points.size == 0:
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header = header
        pointcloud_msg.height = 1
        pointcloud_msg.width = 0
        pointcloud_msg.fields = []
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.point_step = 0
        pointcloud_msg.row_step = 0
        pointcloud_msg.is_dense = True
        pointcloud_msg.data = b""
        return pointcloud_msg

    if colors is not None:
        if colors.shape[1] == 3:
            rgb = (
                (colors[:, 0].astype(np.uint32) << 16)
                | (colors[:, 1].astype(np.uint32) << 8)
                | (colors[:, 2].astype(np.uint32))
            )
        else:
            raise ValueError("Invalid color format")

        structured_array = np.zeros(
            points.shape[0],
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.uint32),
            ],
        )
        structured_array["x"] = points[:, 0]
        structured_array["y"] = points[:, 1]
        structured_array["z"] = points[:, 2]
        structured_array["rgb"] = rgb

        data = structured_array.tobytes()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        point_step = 16
    else:
        structured_array = np.zeros(
            points.shape[0],
            dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        )
        structured_array["x"] = points[:, 0]
        structured_array["y"] = points[:, 1]
        structured_array["z"] = points[:, 2]

        data = structured_array.tobytes()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_step = 12

    pointcloud_msg = PointCloud2()
    pointcloud_msg.header = header
    pointcloud_msg.height = 1
    pointcloud_msg.width = points.shape[0]
    pointcloud_msg.fields = fields
    pointcloud_msg.is_bigendian = False
    pointcloud_msg.point_step = point_step
    pointcloud_msg.row_step = pointcloud_msg.point_step * pointcloud_msg.width
    pointcloud_msg.is_dense = True
    pointcloud_msg.data = data

    return pointcloud_msg
