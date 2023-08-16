import numpy
import numpy as np
import torch
from numpy.matlib import repmat
import cv2
import carla
import math
import queue


def set_sync_mode(world, fix_delta_sec=0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True  # 启用同步模式
    settings.fixed_delta_seconds = fix_delta_sec
    world.apply_settings(settings)


def get_sensor(world, sensor_type: str, attach_actor: carla.Actor, location=(0, 0, 2), rotation=(0, 0, 0)):
    """
    get sensor(depth camera or rgb camera).
    args location(x, y, z) and rotation(yaw, pitch, roll) are relative to the actor it attached to.
    """
    x, y, z = location
    yaw, pitch, roll = rotation

    bp_lib = world.get_blueprint_library()
    sensor_bp = bp_lib.find('sensor.camera.' + sensor_type)
    sensor_bp.set_attribute('fov', '90')
    init_trans = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw, pitch=pitch, roll=roll))
    if sensor_type == 'depth':
        sensor = world.spawn_actor(sensor_bp, init_trans, attach_to=attach_actor,
                                   attachment_type=carla.AttachmentType.Rigid)
    elif sensor_type == 'rgb':
        sensor = world.spawn_actor(sensor_bp, init_trans, attach_to=attach_actor)
    else:
        assert False, 'only depth and rgb camera supported.'

    sensor_data_queue = queue.Queue()
    sensor.listen(sensor_data_queue.put)

    return sensor, sensor_data_queue


def _to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.sensor.Image")
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))

    return array


def _depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = _to_bgra_array(image)
    array = array.astype(numpy.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = numpy.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

    return normalized_depth


def depth_to_local_point_cloud(image, color=None, max_depth=0.9):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    "max_depth" is used to omit the points that are far enough.
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = _depth_to_array(image)

    # (Intrinsic) K Matrix
    k = numpy.identity(3)
    k[0, 2] = image.width / 2.0
    k[1, 2] = image.height / 2.0
    k[0, 0] = k[1, 1] = image.width / (2.0 * math.tan(image.fov * math.pi / 360.0))

    # 2d pixel coordinates
    pixel_length = image.width * image.height
    u_coord = repmat(numpy.r_[image.width-1:-1:-1],
                     image.height, 1).reshape(pixel_length)
    v_coord = repmat(numpy.c_[image.height-1:-1:-1],
                     1, image.width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = numpy.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = numpy.where(normalized_depth > max_depth)
    normalized_depth = numpy.delete(normalized_depth, max_depth_indexes)
    u_coord = numpy.delete(u_coord, max_depth_indexes)
    v_coord = numpy.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = numpy.delete(color, max_depth_indexes, axis=0)

    # pd2 = [u,v,1]
    p2d = numpy.array([u_coord, v_coord, numpy.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = numpy.dot(numpy.linalg.inv(k), p2d)

    p3d *= normalized_depth * far
    p3d = numpy.concatenate((p3d, numpy.ones((1, p3d.shape[1]))))

    return p3d, color


def get_camera2world_matrix(carla_transform: carla.Transform, real_y_axis=False):
    """
    Args:
        carla_transform: Carla.Transform instance, contains carla.Location and carla.Rotation
        real_y_axis: return real y-axis value when setting true. but the view of point cloud in open-3d
            will be reversed in yaw direction.
    Returns:
        a 4x4 rotation & transaction matrix that transforms coords from camera coord-sys to simu-world coord-sys.
    """
    camera2vehicle_matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

    pitch = carla_transform.rotation.pitch / 180.0 * math.pi
    yaw = carla_transform.rotation.yaw / 180.0 * math.pi
    roll = carla_transform.rotation.roll / 180.0 * math.pi
    loc_x = carla_transform.location.x
    loc_y = - carla_transform.location.y
    loc_z = carla_transform.location.z
    sin_y, sin_p, sin_r = math.sin(yaw), math.sin(pitch), math.sin(roll)
    cos_y, cos_p, cos_r = math.cos(yaw), math.cos(pitch), math.cos(roll)

    vehicle2world_matrix = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r + sin_y * cos_r, - cos_y * sin_p * cos_r + sin_y * sin_r, loc_x],
        [-sin_y * cos_p, - sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r + cos_y * sin_r, loc_y],
        [sin_p, -cos_p * sin_r, cos_p * cos_r, loc_z],
        [0.0, 0.0, 0.0, 1.0]
    ])
    if real_y_axis:
        vehicle2world_matrix[1] *= -1

    return vehicle2world_matrix @ camera2vehicle_matrix


def get_world2camera_matrix(carla_transform: carla.Transform, real_y_axis=False):
    # return inverse c2w matrix
    return np.linalg.inv(get_camera2world_matrix(carla_transform, real_y_axis))


class CarlaVirtualObject:

    def __init__(self, object_path):
        pass

    def data2pcd(self, *args, **kwargs):
        pass
