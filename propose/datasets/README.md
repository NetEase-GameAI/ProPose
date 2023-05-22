## Dataset Notes

We illustrate some annotations here:
- **bbox**: The bounding box in the form of `(xmin, ymin, xmax, ymax)`.
- **joint_cam_X**: `(X, 3)`, The 3D positions of X-joints in the camera coordinate system, with mm as unit.
- **joint_vis_X**: `(X, 3)`, The visibility of joint_cam_X.
- **joint_img_X**: `(X, 3)`, The 2D positions and relative depths of joints in the form of `uvd`. Especially, for `X=11`, the shape is `(X, 3, 2)` with visibility.
- **theta**: `(24, 3)`, SMPL pose parameters.
- **beta**: `(10,)`, SMPL shape parameters.
- **root_cam**: `(3,)`, The 3D root position of human in the camera coordinate system.
- **f**: `(2,)`, optional, camera focal length. For general scenes, we use a default focal length, where the camera annotation can thus be ignored.
- **c**: `(2,)`, optional, camera principal point.

The number of joints X can be:
- **29**: 24 SMPL joints + 5 leaf joints (used by [HybrIK](https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/models/layers/smpl/lbs.py#L351)).
- **17**: Human3.6M format for evaluation.
- **11**: 5 face joints and 6 foot joints defined in SMPLX (see [convert.py](./convert.py#L54))

The details of further processing can be found in [wrapper](../utils/wrapper/smpl3dmetric.py).
