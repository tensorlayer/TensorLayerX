import tensorflow as tf

__all__ = [
    'box_iou',
    'nms',
    'box_area',
]

def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Parameters
    ----------
    boxes :(Tensor[N, 4])
        boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns
    -------
    area : area (Tensor[N])
        area for each box
    """

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

    tl = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = tf.clip_by_value(rb - tl, clip_value_min = 0, clip_value_max = rb - tl)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    """Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    Parameters
    ----------
    boxes : Tensor
        Tensor[N, 4]
    scores : Tensor
        Tensor[N], scores for each one of the boxes
    iou_threshold : float
        discards all overlapping boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
    """

    return tf.image.non_max_suppression(boxes = boxes, scores = scores, max_output_size=50, iou_threshold=iou_threshold)
