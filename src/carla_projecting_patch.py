import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import queue
import carla
import numpy
import numpy as np
import cv2
import math

from typing import *
from utils import get_camera2world_matrix, get_world2camera_matrix, set_sync_mode, get_sensor
from utils import CarlaVirtualObject, depth_to_local_point_cloud, _depth_to_array

# arguments
import argparse
parser = argparse.ArgumentParser(
    prog='carla projecting patch',
    description='project patch in carla and display in rgb camera output.'
)
parser.add_argument('--host', metavar='H',    default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
parser.add_argument('--port', '-p',           default=2000, type=int, help='TCP port to listen to (default: 2000)')
parser.add_argument('--tm_port',              default=8000, type=int, help='Traffic Manager Port (default: 8000)')
parser.add_argument('--top-view',             default=True, help='Setting spectator to top view on ego car')
parser.add_argument('--map',                  default='Town10HD', help='Town Map')
parser.add_argument('--save_data_path',       default='../PointCloud/', help='Path to save point cloud files')
parser.add_argument('--sync_mode',            default=True, help='enable sync mode')

parser.add_argument('--spawn_point',          default=5, type=int, help='spawn point index.')
parser.add_argument('--object_path',          default='../patch.npy', help='Path to object to project')
parser.add_argument('--relative_pos',         default=True,
                    help='when True, location settings of projector camera will be relative to ego vehicle')
parser.add_argument('--x',                    default=0, type=float, help='x coordinates of projector camera')
parser.add_argument('--y',                    default=0, type=float, help='y coordinates of projector camera')
parser.add_argument('--z',                    default=10, type=float, help='z coordinates of projector camera')
parser.add_argument('--yaw',                  default=10, type=float, help='yaw angle of projector camera')
parser.add_argument('--pitch',                default=5, type=float, help='pitch angle of projector camera')
parser.add_argument('--roll',                 default=0, type=float, help='roll angle of projector camera')
parser.add_argument('--fov',                  default=15, type=int, help='fov of projector camera')
parser.add_argument('--patch_H',              default=560, type=int, help='height of patch object')
parser.add_argument('--patch_W',              default=560, type=int, help='width of patch object')

arguments = parser.parse_args()


class CarlaPatch2Prj(CarlaVirtualObject):
    """
    load patch to project in carla world, get point cloud in return value.
    """
    def __init__(self, data_filepath: str):
        super().__init__(data_filepath)
        patch_array = np.load(data_filepath)
        self.patch_data = patch_array

    def data2pcd(self, depth_data: carla.Image, prj_depth_camera: carla.Sensor):
        patch_p3d, patch_color = depth_to_local_point_cloud(depth_data, self.patch_data, max_depth=0.9)
        c2w_mat = get_camera2world_matrix(prj_depth_camera.get_transform())
        patch_p3d = (c2w_mat @ patch_p3d)[:3]

        return patch_p3d, patch_color


def get_patch_projector(world, location: Tuple, rotation: Tuple, fov: float, patch_size: Tuple, relative_pos=True):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

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

        # generate vehicles
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[5])
        finally_tasks.put({'func': vehicle.destroy, 'args': None, 'description': 'destroy vehicle'})

        # set traffic manager and autopilot
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        rgb_camera, rgb_image_queue = get_sensor(world, 'rgb', vehicle, (0, 0, 2), (0, 0, 0))
        finally_tasks.put({'func': rgb_camera.destroy, 'args': None, 'description': 'destroy rgb camera'})
        depth_camera, depth_image_queue = get_sensor(world, 'depth', vehicle, (0, 0, 2), (0, 0, 0))
        finally_tasks.put({'func': depth_camera.destroy, 'args': None, 'description': 'destroy depth camera'})

        prj_depth_camera, prj_depth_image_queue = get_patch_projector(
            world, (arguments.x, arguments.y, arguments.z), (arguments.yaw, arguments.pitch, arguments.roll),
            arguments.fov, (arguments.patch_W, arguments.patch_H), arguments.relative_pos
        )

        world.tick()

        prj_depth_image = prj_depth_image_queue.get()
        carla_patch = CarlaPatch2Prj(arguments.object_path)
        object_p3d, object_color = carla_patch.data2pcd(prj_depth_image, prj_depth_camera)
        print('project patch to world...')
        prj_depth_camera.destroy()
        print('destroy projector')
        # display RGB camera output by cv2 window
        cv2.namedWindow('RGB Camera Output', cv2.WINDOW_AUTOSIZE)

        while cv2.waitKey(1) != ord('q'):
            world.tick()

            depth_image = depth_image_queue.get()
            carla_rgb_image = rgb_image_queue.get()
            rgb_image = np.reshape(np.copy(carla_rgb_image.raw_data), (carla_rgb_image.height, carla_rgb_image.width, 4))

            if arguments.top_view:
                # add a spector to see how our vehicle move
                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            k = numpy.identity(3)
            k[0, 2] = carla_rgb_image.width / 2.0
            k[1, 2] = carla_rgb_image.height / 2.0
            k[0, 0] = k[1, 1] = carla_rgb_image.width / (2.0 * math.tan(carla_rgb_image.fov * math.pi / 360.0))

            camera_transform = rgb_camera.get_transform()

            world2camera_matrix = get_world2camera_matrix(camera_transform)

            p2d = k @ (world2camera_matrix @ np.concatenate([object_p3d, np.ones((1, object_p3d.shape[1]))]))[:3]
            p2d[0] /= p2d[2]
            p2d[1] /= p2d[2]
            # convert to int64 as index
            p2d = numpy.array(p2d + 0.5, dtype=np.int64)
            # box mask
            box_mask = (p2d[0] >= 0) & (p2d[1] >= 0) & \
                       (p2d[0] < carla_rgb_image.width) & \
                       (p2d[1] < carla_rgb_image.height)

            p2d = box_mask * p2d + ~box_mask * np.zeros_like(p2d[0])

            # depth mask
            # didn't work. todo: fix it
            # far = 1000.0
            # foreground_depth = _depth_to_array(depth_image) * far  # [H, W]
            # rgb_camera_loc = rgb_camera.get_transform().location
            # p3d_depth = (object_p3d - np.array([[rgb_camera_loc.x], [rgb_camera_loc.y], [rgb_camera_loc.z]])) ** 2
            # p3d_depth = np.sqrt(np.sum(p3d_depth, axis=0))  # (3, N) -> (1, N)
            #
            # depth_mask = (foreground_depth[-p2d[1], -p2d[0]] - p3d_depth) < 0
            # p2d = depth_mask * p2d + ~depth_mask * np.zeros_like(p2d[0])

            c_point = object_p3d[:, object_p3d.shape[1] // 2]
            ray = carla.Location(c_point[0], - c_point[1], c_point[2]) - rgb_camera.get_transform().location
            forward_vec = rgb_camera.get_transform().get_forward_vector()
            # to make sure patch is in the front of vehicle and display it.

            if forward_vec.dot(ray) > 0:
                # replace pixels in camera output with patch pixels
                rgb_image[-p2d[1], -p2d[0], :3] = np.array(object_color * 255, dtype=np.uint8)
            cv2.imshow('RGB Camera Output', rgb_image)

    finally:
        # do destroy/clear stuffs at finally
        while not finally_tasks.empty():
            finally_task = finally_tasks.get()
            task, args, des = finally_task['func'], finally_task['args'], finally_task['description']
            task(args) if args is not None else task()
            print(des)


if __name__ == '__main__':
    main()
