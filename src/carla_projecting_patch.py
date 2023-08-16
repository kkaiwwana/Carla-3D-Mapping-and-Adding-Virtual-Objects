import queue
import carla
import numpy
import numpy as np
import cv2
import open3d as o3d
import math
import torch
import time

from typing import *
from utils import get_camera2world_matrix, get_world2camera_matrix, set_sync_mode, get_sensor
from utils import CarlaVirtualObject, depth_to_local_point_cloud


# arguments
import argparse
parser = argparse.ArgumentParser(
    prog='carla 3d-mapping',
    description='carla 3d-mapping by using depth and rgb camera'
)
parser.add_argument('--host', metavar='H',    default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p',           default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port',              default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--top-view',             default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map',                  default='Town10HD', help='Town Map')
parser.add_argument('--object_path',          default='../patch.pt', help='Path to object to project')
parser.add_argument('--save_data_path',       default='../PointCloud/', help='Path to save point cloud files')
parser.add_argument('--real_y_axis',          default=False, help='Get real y axis value in point cloud')
parser.add_argument('--sync_mode',            default=True, help='enable sync mode')

parser.add_argument('--relative_pos',         default=True, help='enable sync mode')
parser.add_argument('--x',                    default=0, type=float, help='enable sync mode')
parser.add_argument('--y',                    default=0, type=float, help='enable sync mode')
parser.add_argument('--z',                    default=40, type=float, help='enable sync mode')
parser.add_argument('--yaw',                  default=0, type=float, help='enable sync mode')
parser.add_argument('--pitch',                default=0, type=float, help='enable sync mode')
parser.add_argument('--roll',                 default=0, type=float, help='enable sync mode')
parser.add_argument('--fov',                  default=15, type=int, help='enable sync mode')
parser.add_argument('--patch_H',              default=128, type=int, help='enable sync mode')
parser.add_argument('--patch_W',              default=128, type=int, help='enable sync mode')

arguments = parser.parse_args()


class CarlaPatch2Prj(CarlaVirtualObject):
    """
    load patch to project in carla world, get point cloud in return value.
    """
    def __init__(self, data_filepath: torch.Tensor):
        super().__init__(data_filepath)
        patch_tensor = torch.load(data_filepath, map_location='cpu')
        self.patch_data = patch_tensor.permute((1, 2, 0)).detach().numpy()

    def data2pcd(self, depth_data: carla.Image, prj_depth_camera: carla.Sensor, real_y_axis):
        patch_p3d, patch_color = depth_to_local_point_cloud(depth_data, self.patch_data, max_depth=0.9)
        c2w_mat = get_camera2world_matrix(prj_depth_camera.get_transform(), real_y_axis=real_y_axis)
        patch_p3d = (c2w_mat @ patch_p3d)[:3]

        return patch_p3d, patch_color


def get_patch_projector(world, location: Tuple, rotation: Tuple, fov: float, patch_size: Tuple, relative_pos=True):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # prj_vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    # prj_location = spawn_points[5]
    # prj_vehicle = world.try_spawn_actor(prj_vehicle_bp, prj_location)

    prj_depth_camera_bp = bp_lib.find('sensor.camera.depth')
    prj_depth_camera_bp.set_attribute('fov', str(fov))
    prj_depth_camera_bp.set_attribute('image_size_x', str(patch_size[0]))
    prj_depth_camera_bp.set_attribute('image_size_y', str(patch_size[1]))

    if not relative_pos:
        # when location, rotation passed is relative(relative to vehicle) position
        x, y, z = location
        yaw, pitch, roll = rotation

    else:
        # convert absolute(relative to world) location, rotation to relative position
        relative_trans = spawn_points[5]
        ve_x, ve_y, ve_z = relative_trans.location.x, relative_trans.location.y, relative_trans.location.z
        ve_ya, ve_pi, ve_ro = relative_trans.rotation.yaw, relative_trans.rotation.pitch, relative_trans.rotation.roll
        x, y, z = location[0] + ve_x, location[1] + ve_y, location[2] + ve_z
        yaw, pitch, roll = rotation[0] + ve_ya, rotation[1] + ve_pi, rotation[2] + ve_ro

    prj_depth_camera_init_trans = \
        carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

    prj_depth_camera = world.spawn_actor(prj_depth_camera_bp, prj_depth_camera_init_trans, attach_to=None,
                                         attachment_type=carla.AttachmentType.Rigid)

    prj_depth_image_queue = queue.Queue()
    prj_depth_camera.listen(prj_depth_image_queue.put)

    return prj_depth_camera, prj_depth_image_queue


def main():
    finally_tasks = queue.Queue()

    client = carla.Client(arguments.host, arguments.port)
    world = client.load_world(arguments.map)
    default_settings = world.get_settings()
    finally_tasks.put({'func': world.apply_settings, 'args': default_settings, 'description': 'reset world to default'})

    try:
        # set sync mode. it's recommend to enable it.
        if arguments.sync_mode:
            set_sync_mode(world)

        traffic_manager = client.get_trafficmanager(arguments.tm_port)
        traffic_manager.set_synchronous_mode(True)
        finally_tasks.put(
            {'func': traffic_manager.set_synchronous_mode, 'args': False, 'description': 'disable tm sync mode'}
        )

        bp_lib = world.get_blueprint_library()

        # 生成车辆
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[5])
        finally_tasks.put({'func': vehicle.destroy, 'args': None, 'description': 'destroy vehicle'})

        # set traffic manager and autopilot
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        rgb_camera, rgb_image_queue = get_sensor(world, 'rgb', vehicle, (0, 0, 2), (0, 0, 0))
        finally_tasks.put({'func': rgb_camera.destroy, 'args': None, 'description': 'destroy rgb camera'})
        depth_camera, depth_image_queue = get_sensor(world, 'depth', vehicle)
        finally_tasks.put({'func': depth_camera.destroy, 'args': None, 'description': 'destroy depth camera'})

        prj_depth_camera, prj_depth_image_queue = get_patch_projector(
            world, (arguments.x, arguments.y, arguments.z), (arguments.yaw, arguments.pitch, arguments.roll),
            arguments.fov, (arguments.patch_W, arguments.patch_H), arguments.relative_pos
        )

        world.tick()

        rgb_image = rgb_image_queue.get()
        prj_depth_image = prj_depth_image_queue.get()
        carla_patch = CarlaPatch2Prj(arguments.object_path)
        object_p3d, object_color = carla_patch.data2pcd(prj_depth_image, prj_depth_camera, arguments.real_y_axis)
        prj_depth_camera.destroy()
        # prj_vehicle.destroy()

        # 在 OpenCV 的显示窗口中显示图像
        cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)

        while True:
            world.tick()

            depth_image = depth_image_queue.get()
            carla_rgb_image = rgb_image_queue.get()

            k = numpy.identity(3)
            k[0, 2] = carla_rgb_image.width / 2.0
            k[1, 2] = carla_rgb_image.height / 2.0
            k[0, 0] = k[1, 1] = carla_rgb_image.width / (2.0 * math.tan(carla_rgb_image.fov * math.pi / 360.0))

            rgb_image = np.reshape(np.copy(carla_rgb_image.raw_data), (carla_rgb_image.height, carla_rgb_image.width, 4))

            # We here add a spectator to watch how our ego vehicle will move
            if arguments.top_view:
                spectator = world.get_spectator()
                transform = vehicle.get_transform()  # we get the transform of vehicle
                spectator.set_transform(
                    carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            camera_transform = rgb_camera.get_transform()
            c2w = get_camera2world_matrix(camera_transform, real_y_axis=False)
            world2camera_matrix = np.linalg.inv(c2w)

            ##################

            camera2vehicle_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

            pitch = camera_transform.rotation.pitch / 180.0 * math.pi
            yaw = camera_transform.rotation.yaw / 180.0 * math.pi
            roll = camera_transform.rotation.roll / 180.0 * math.pi
            loc_x = camera_transform.location.x
            loc_y = - camera_transform.location.y
            loc_z = camera_transform.location.z

            rx = np.array([
                [1, 0, 0], [0, math.cos(roll), - math.sin(roll)], [0, math.sin(roll), math.cos(roll)],
            ], dtype=np.float64)
            ry = np.array([
                [math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [- math.sin(pitch), 0, math.cos(pitch)]
            ], dtype=np.float64)
            rz = np.array([
                [math.cos(yaw), - math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]
            ], dtype=np.float64)

            w2c = camera2vehicle_matrix @ rx @ ry @ rz

            ##################

            p2d = k @ (w2c @ (object_p3d[:3] - np.array(([[loc_x], [loc_y], [loc_z]]))))[:3]
            p2d[0] /= p2d[2]
            p2d[1] /= p2d[2]
            p2d = numpy.array(p2d + 0.5, dtype=np.int64)

            mask = (p2d[0] >= 0) & (p2d[1] >= 0) & (p2d[0] < carla_rgb_image.width) & (p2d[1] < carla_rgb_image.height)
            zeros_indices = np.zeros_like(p2d[0])
            p2d = mask * p2d + ~mask * zeros_indices

            ray = carla.Location(object_p3d[:, 0][0].item(), - object_p3d[:, 0][1].item(), object_p3d[:, 0][2].item()) - rgb_camera.get_transform().location
            forward_vec = rgb_camera.get_transform().get_forward_vector()

            if forward_vec.dot(ray) > 0:
                # replace pixels in camera output with patch pixels
                rgb_image[-p2d[1], -p2d[0], :3] = np.array(object_color * 255, dtype=np.uint8)

            cv2.imshow('ImageWindowName', rgb_image)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        # do destroy/clear stuffs at finally
        while not finally_tasks.empty():
            finally_task = finally_tasks.get()
            task, args, des = finally_task['func'], finally_task['args'], finally_task['description']
            task(args) if args is not None else task()
            print(des)


if __name__ == '__main__':
    main()
