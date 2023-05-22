import numpy as np


def bbox_xywh_to_xyxy(xywh):
    """ Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax).

    Args:
    - xywh (list, tuple or np.ndarray): The bbox in format (x, y, w, h). 
        If input is np.ndarray `(N, 4)`, output has the same shape `(N, 4)`.

    Returns:
    - tuple or np.ndarray: The converted bboxes in format (xmin, ymin, xmax, ymax).
    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(f'Bbox must have 4 elements, given {len(xywh)}')
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(f'Bbox must have n * 4 elements, given {xywh.shape}')
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(f'Input must be a list, tuple or np.ndarray, given {type(xywh)}')


def bbox_xyxy_to_xywh(xyxy):
    """ Convert bounding boxes from format (xmin, ymin, xmax, ymax) to (x, y, w, h).

    Args:
    - xyxy (list, tuple or np.ndarray): The bbox in format (xmin, ymin, xmax, ymax).
        If input is np.ndarray `(N, 4)`, output has the same shape `(N, 4)`.

    Returns:
    - tuple or np.ndarray: The converted bboxes in format (x, y, w, h).
    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(f'Bbox must have 4 elements, given {len(xyxy)}')
        x1, y1 = xyxy[0], xyxy[1]
        w, h = xyxy[2] - x1 + 1, xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(f'Bbox must have n * 4 elements, given {xyxy.shape}')
        return np.hstack((xyxy[:, :2], xyxy[:, 2:4] - xyxy[:, :2] + 1))
    else:
        raise TypeError(f'Input must be a list, tuple or np.ndarray, given {type(xyxy)}')


def bbox_clip_xyxy(xyxy, width, height):
    """ Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.
        All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Args:
    - xyxy (list, tuple or np.ndarray): The bbox in format (xmin, ymin, xmax, ymax).
            If input is np.ndarray `(N, 4)`, output has the same shape `(N, 4)`.
    - width (int or float): Boundary width.
    - height (int or float): Boundary height.

    Returns:
    - tuple or np.ndarray: The clipped xyxy.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(f'Bbox must have 4 elements, given {len(xyxy)}')
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(f'Bbox must have n * 4 elements, given {xyxy.shape}')
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(f'Input must be a list, tuple or np.ndarray, given {type(xyxy)}')


def bbox_xywh_to_cs(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """ Convert bounding boxes from format (x, y, w, h) to (center, scale). """

    pixel_std = 1
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale


def bbox_cs_to_xyxy(center, scale):
    """ Convert bounding boxes from format (center, scale) to (xmin, ymin, xmax, ymax). """

    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox


def calc_iou_xyxy(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    box1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    box2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou


def get_max_iou_box(det_output, prev_bbox, thrd=0.9):
    """ Get the current bbox with max IOU over the previous bbox. """

    max_score = 0
    max_bbox = None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        iou = calc_iou_xyxy(prev_bbox, bbox)
        iou_score = float(score) * iou
        if float(iou_score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = iou_score

    return max_bbox


def get_one_box(det_output, thrd=0.9):
    """ Get one box from detection results. """
    
    max_area = 0
    max_bbox = None
    if thrd < 0.3:
        return None
    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox