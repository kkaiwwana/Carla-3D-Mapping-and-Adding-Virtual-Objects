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
import open3d as o3d
import time
from tqdm import tqdm

from utils import depth_to_local_point_cloud, get_camera2world_matrix

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
parser.add_argument('--sampling_per_N_frames',default=40, type=int, help='sampling once after N frames(default: 40)')
parser.add_argument('--sync_mode',            default=True, help='enable sync mode')

arguments = parser.parse_args()


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


def main(*args, **kwargs):
    point_cloud, point_cloud_colors = [], []
    point_list = o3d.geometry.PointCloud()
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

        # generate vehicle
        vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        finally_tasks.put({'func': vehicle.destroy, 'args': None, 'description': 'destroy vehicle'})

        # set traffic manager and autopilot
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        rgb_camera, rgb_image_queue = get_sensor(world, 'rgb', vehicle)
        finally_tasks.put({'func': rgb_camera.destroy, 'args': None, 'description': 'destroy rgb camera'})
        depth_camera, depth_image_queue = get_sensor(world, 'depth', vehicle)
        finally_tasks.put({'func': depth_camera.destroy, 'args': None, 'description': 'destroy depth camera'})

        # world.tick()
        # rgb_image = rgb_image_queue.get()
        # # 将原始数据重新整形为 RGB 数组
        # rgb_img = np.reshape(np.copy(rgb_image.raw_data), (rgb_image.height, rgb_image.width, 4))

        # 在 OpenCV 的显示窗口中显示图像
        cv2.namedWindow('VehiclePerspective', cv2.WINDOW_AUTOSIZE)
        finally_tasks.put({'func': cv2.destroyAllWindows, 'args': None, 'description': 'destroy opencv windows'})
        # cv2.imshow('VehiclePerspective', rgb_img)
        # cv2.moveWindow('VehiclePerspective', 0, 600)
        # cv2.waitKey(1)

        # use open3D to visualize point cloud data
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Display Point Cloud", width=960, height=540, left=0, top=60)
        finally_tasks.put({'func': vis.destroy_window, 'args': None, 'description': 'destroy open3d window'})
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        frame_cnt = -1

        while cv2.waitKey(1) != ord('q'):
            world.tick()
            frame_cnt += 1

            depth_image = depth_image_queue.get()
            rgb_image = rgb_image_queue.get()
            rgb_image = np.reshape(np.copy(rgb_image.raw_data), (rgb_image.height, rgb_image.width, 4))
            cv2.imshow('VehiclePerspective', rgb_image)

            if arguments.top_view:
                # add a spector to see how our vehicle move
                spectator = world.get_spectator()
                transform = vehicle.get_transform()
                spectator.set_transform(
                    carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

            if frame_cnt % arguments.sampling_per_N_frames == 0:

                camera_transform = depth_camera.get_transform()
                # get translation and rotation matrix
                # WATCH OUT! Use carla.Transform().get_matrix() will cause mistakes.
                camera2world_matrix = get_camera2world_matrix(camera_transform, real_y_axis=arguments.real_y_axis)

                # convert p2d to p3d(local) and convert color RGB <-> BGR
                p3d, color = depth_to_local_point_cloud(depth_image, rgb_image[..., [2, 1, 0]], max_depth=0.6)
                p3d = (camera2world_matrix @ p3d)[:3]

                # save point cloud and color data
                point_cloud.append(p3d)
                point_cloud_colors.append(color)
                # update point cloud
                point_list.points.extend(o3d.utility.Vector3dVector(p3d.transpose()))
                point_list.colors.extend(o3d.utility.Vector3dVector(color / 255.0))

                vis.add_geometry(point_list) if frame_cnt == 0 else vis.update_geometry(point_list)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.005)

    finally:
        # do destroy/clear stuffs at finally
        while not finally_tasks.empty():
            finally_task = finally_tasks.get()
            task, args, des = finally_task['func'], finally_task['args'], finally_task['description']
            task(args) if args is not None else task()
            print(des)
        # try save point cloud files.
        try:
            numpy.save(arguments.save_data_path + 'world_point_cloud.npy',
                       np.concatenate(point_cloud, axis=1).transpose())
        except FileNotFoundError:
            numpy.save('./world_point_cloud.npy', np.concatenate(point_cloud, axis=1).transpose())
            print(f'No such dir. Check the directory you passed.'
                  f'You point cloud data is temporary save to working directory.')
        else:
            print(f'point cloud data is successfully saved in {arguments.save_data_path}world_point_cloud.npy')

        try:
            numpy.save(arguments.save_data_path + 'world_point_cloud_colors.npy',
                       np.concatenate(point_cloud_colors, axis=0))
        except FileNotFoundError:
            numpy.save('./world_point_cloud_color.npy', np.concatenate(point_cloud_colors, axis=0))
            print(f'No such dir. Check the directory you passed.'
                  f'You point cloud color data is temporary save to working directory.')
        else:
            print(f'point cloud color data is successfully saved in '
                  f'{arguments.save_data_path}world_point_cloud_color.npy')

        # do some down_sample() if you need
        # point_list.farthest_point_down_sample(Args), this is for demonstration
        o3d.io.write_point_cloud(arguments.save_data_path + 'o3d_world_point_cloud.pcd', point_list)


if __name__ == '__main__':
    main()