
## Discussion
The current code, mainly the keypoint branch, is somewhat different from the description in the paper. In the paper, we use the `uvd` format following [HybrIK](https://github.com/Jeff-sjtu/HybrIK), i.e., the 2D keypoints in the image space and the relative depth to the root joint. To tackle the truncation better, we here adopt the `xyz` format similar to [MeTRAbs](https://github.com/isarandi/metrabs), i.e., the 3D joint poistions in the metric space. The former is more accurate when the person is not truncated, while the latter is more robust for various cases. 

There are still some limitations that need to be addressed:
- The foot is not accurate enough even if foot keypoints are additionally supervised, same for camera. 
- Sampling can generate diverse human proposals in adjacent joints, but there is no guarantee that the whole person is not twisted, since the ancestor joints are not taken into consideration in probabilistic modeling.