# Carla 3D-Mapping and Adding Virtual Objects
3D-Mapping in Carla by using Depth & RGB camera and adding virtual objects(project patch to carla world in this demo) to camera output
## Quick Start
### clone this repo & run carla simu
```bash
git clone https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects.git
cd [YOUR-PATH-TO-CARLA]
.\CarlaUE4
```
### carla 3d-mapping
```bash
cd carla_3d-mapping_and_adding_virtual_objects/src
python carla_3d_mapping.py --map Town10HD
```
### point cloud visualizing
```bash
python visualize_point_cloud.py --files [YOUR-POINT-CLOUD-FILE-PATH]
```
### patch projecting
```bash
python carla_projecting_patch.py --z 40 --pitch 0
```

## Carla 3D-Mapping
![carla_3d_mapping](https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects/blob/main/IMG/carla_pc.png)
## Add Virtual Object
![project_patch](https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects/blob/main/IMG/project_patch.png)
## Motivation
#### I didn't really find some work like this (I see some Carla Mapping by using LiDAR, no color and C++ project). There must be some reason that stopped them to do something like it and I think I find that: there might be a BUG in Carla coordinates transform API.
#### Bassicly, some code dealing with coordinates transform went wrong in carla, which leads to some problem that make transform coordinates in Carla between Camera-Coord-Sys and World-Coord-Sys went wrong and then make merging multiple point cloud to one impossible. There are some related issues: [#553 Transform](https://github.com/carla-simulator/carla/issues/553) | [#3051 Merging multiple point cloud from depth camera](https://github.com/carla-simulator/carla/issues/3051) | [#6435 Merge/align point clouds from different perspectives](https://github.com/carla-simulator/carla/issues/6435))
#### I implemented that part of code by my own and made it working and looks good to me. TODO: I'll open a Issue/PR in Carla later to see what happend actually.
