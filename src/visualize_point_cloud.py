import open3d as o3d
import argparse

parser = argparse.ArgumentParser(prog='visualize point cloud')

parser.add_argument('--f', '--files',          dest='files', nargs='+', help="single or list of *.pcd open3d files path")
parser.add_argument('--point_size',            default=1.0, type=float, help='point size of point cloud')
parser.add_argument('--window_size',           default=[540, 940], type=list, help='window size, [H, W]')
parser.add_argument('--window_loc',            default=[100, 100], type=list, help='window location, [Left, Top]')
parser.add_argument('--window_name',           default='Point Cloud', help='window name')
arguments = parser.parse_args()


def main(*args, **kwargs):

    point_clouds = [o3d.io.read_point_cloud(file) for file in arguments.files]

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=arguments.window_name,
        width=arguments.window_size[1],
        height=arguments.window_size[0],
        left=arguments.window_loc[0],
        top=arguments.window_loc[1]
    )
    # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = arguments.point_size
    vis.get_render_option().show_coordinate_frame = True

    for i, point_cloud in enumerate(point_clouds):
        vis.add_geometry(point_cloud) if i == 0 else vis.add_geometry(point_cloud)

    vis.poll_events()
    vis.update_renderer()
    vis.run()


if __name__ == '__main__':
    main()
