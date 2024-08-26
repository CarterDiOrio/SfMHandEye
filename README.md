# SfMHandEye
Corresponding portfolio post: www.cdiorio.dev/projects/camera-calibration/#targetless-hand-eye-calibration

This project performs Structure from Motion based targetless hand eye calibration. It only uses the robot arms motion for scale currently. The views
in the first group are used to establish scale, so at least two views are needed in the first group.

### How to use
```
calibrate <data_directory> <fx> <cx> <fy> <cy>
```
Where:
- data_file: the filepath to the data json and images
- fx: the x focal length
- fy: the y focal length
- cx: the center of projection x coordinate
- cy: the center of projection y coorindate

Assumes that the images are already undistorted or can be approximated as a pinhole model.


### Dependencies
- Ceres
- OpenCV
- OpenMVG
- Sophus 
- nlohmann JSON

### Data JSON Scheme

The idea is that cameras within each camera group share some property. The one that is currently used is that cameras within each group under go only
translation relative to other members of the group.

```
DataJson {
  groups: List[CameraGroup]
}

CameraGroup {
  cameras: List[Camera]
}

Camera {
  image: str
  pose: [x, y, z, rot X, rot Y, rot Z] in degrees
}
```