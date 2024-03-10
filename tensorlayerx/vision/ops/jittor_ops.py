import jittor as jt
import jittor.transform
import numpy as np
__all__ = [
    'box_iou',
    'nms',
    'box_area',
]

def box_area(boxes):
    if len(boxes.shape) == 1:
        boxes = boxes[None, :]
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Parameters
    ----------
    boxes1 : Tensor
        Tensor[N, 4]
    boxes2 : Tensor
        Tensor[M, 4]

    Returns
    -------
    iou: Tensor
        Tensor[N, M],the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2

    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    tl = jt.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = jt.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = np.clip(rb - tl, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
    

def nms(boxes, scores, iou_threshold):
    keep = []
    idx = jt.argsort(scores)
    while idx.size > 0:
        if idx.size == 1:
            i = idx[0]
            keep.append(i)
            break
        else:
            max_score_index = idx[-1]
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)
            idx = idx[:-1]
            other_boxes = boxes[idx]
            ious = box_iou(max_score_box, other_boxes)
            idx = idx[ious[0] <= iou_threshold]

    keep = jittor.transform.to_tensor(keep)
    keep = jt.flatten(keep)
    return keep
