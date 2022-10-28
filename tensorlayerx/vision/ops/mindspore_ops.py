import mindspore as ms

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
    op =  ms.ops.IOU(mode='iou')
    return op(boxes2, boxes1)

def nms(boxes, scores, iou_threshold):
    keep = []
    idx = ms.ops.sort(scores)[1].asnumpy()
    while idx.size > 0:
        if idx.size == 1:
            i = idx[0]
            keep.append(i)
            break
        else:
            max_score_index = int(idx[-1])
            max_score_box = boxes[max_score_index][None, :]
            keep.append(max_score_index)
            idx = idx[:-1]
            other_boxes = boxes[ms.Tensor(idx)]
            ious = box_iou(max_score_box, other_boxes)
            idx = ms.ops.masked_select(ms.Tensor(idx), ms.Tensor(ious[0] <= iou_threshold))
            idx = idx.asnumpy()

    keep = ms.Tensor(keep)
    return keep
