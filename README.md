# Carla 3D-Mapping and Adding Virtual Objects in world
3D-Mapping in Carla by using Depth & RGB camera and adding virtual objects(project patch to carla world in this demo) to camera output
## Quick Start
### *1.clone this repo & run carla simu*
```bash
git clone https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects.git
cd [YOUR-PATH-TO-CARLA]
.\CarlaUE4
```
### *carla 3d-mapping*
```bash
cd carla_3d-mapping_and_adding_virtual_objects/src
python carla_3d_mapping.py --map Town10HD
```
### *point cloud visualizing*
```bash
python visualize_point_cloud.py --files [YOUR-POINT-CLOUD-FILE-PATH]
```
### *patch projecting*
```bash
python carla_projecting_patch.py --z 40 --pitch 0
```

## Carla 3D-Mapping
![carla_3d_mapping](https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects/blob/main/IMG/carla_pc.png)
## Add Virtual Object
![project_patch](https://github.com/iaoqian/carla_3d-mapping_and_adding_virtual_objects/blob/main/IMG/project_patch.png)
