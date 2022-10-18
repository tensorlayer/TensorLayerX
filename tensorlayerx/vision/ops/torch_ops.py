import torchvision

__all__ = [
    'box_iou',
    'nms',
    'box_area',
]

def box_area(boxes):

    return torchvision.ops.box_area(boxes)

def box_iou(boxes1, boxes2):
    return torchvision.ops.box_iou(boxes1, boxes2)


def nms(boxes, scores, iou_threshold):
    return torchvision.ops.nms(boxes, scores, iou_threshold)
