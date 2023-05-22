import cv2
import numpy as np


def vis_bbox(image, bbox):
    x1, y1, x2, y2 = bbox

    bbox_img = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

    return bbox_img


def vis_2d_jts(image, pts, weights=None, pairs=None, color=(0, 255, 0)):
    if weights is None:
        weights = np.ones_like(pts)
    
    # joints
    for j in range(len(pts)):
        if weights[j, 0] > 0 and weights[j, 1] > 0:
            x, y = pts[j, :2]
            image = cv2.circle(image, (int(x), int(y)), 3, color, 3)

    # bone
    if pairs is not None:
        for pair in pairs:
            p1, p2 = pair
            if weights[p1, 0] > 0 and weights[p1, 1] > 0 and weights[p2, 0] > 0 and weights[p2, 1] > 0:
                x1, y1 = pts[p1, :2]
                x2, y2 = pts[p2, :2]
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=3)

    return image


def vis_vertices(vertices, renderer, c, img=None, color_id=0, 
                 cam_rt=np.zeros(3), cam_t=np.zeros(3)):
    """ Render SMPL to image. 
    
    Args:
    - vertices: (V, 3), translated vertices.
    - c: (2,), principal point.
    - img: RGB image as background. if None, then use white background.
    """

    rend_img_overlay = renderer(
        vertices, princpt=c, img=img, do_alpha=True, 
        color_id=color_id, cam_rt=cam_rt, cam_t=cam_t)

    rendered_img = rend_img_overlay[:, :, :3].astype(np.uint8)

    return rendered_img
