import paddle

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
    if len(boxes1.shape) == 1:
        boxes1 = boxes1[None, :]
    if len(boxes2.shape) == 1:
        boxes2 = boxes2[None, :]
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = paddle.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = paddle.clip((rb - lt), min = 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    keep = []
    idx = paddle.argsort(scores)
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

    keep = paddle.to_tensor(keep)
    keep = paddle.flatten(keep)
    return keep
