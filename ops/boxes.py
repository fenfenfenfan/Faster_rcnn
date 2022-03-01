import torch
import torchvision
from typing import Tuple
from torch import Tensor


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int,int]) -> Tensor
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(
            boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(
            boxes_x, torch.tensor(width,
                                  dtype=boxes.dtype,
                                  device=boxes.device))
        boxes_y = torch.max(
            boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(
            boxes_y,
            torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    # stack->reshape: make clipped_boxes[xmin,ymin,xmax,ymax]
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    keep = torch.where(keep)[0]

    return keep


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()

    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # x1 = torch.where(boxes1[:,0]>boxes2[:,0],boxes1[:,0],boxes2[:,0])
    # y1 = torch.where(boxes1[:,1]>boxes2[:,1],boxes1[:,1],boxes2[:,1])
    # x2 = torch.where(boxes1[:,2]<boxes2[:,2],boxes1[:,2],boxes2[:,2])
    # y2 = torch.where(boxes1[:,3]<boxes2[:,3],boxes1[:,3],boxes2[:,3])

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes2[:, None, 2:], boxes2[:, 2:])

    wh = (right_bottom - left_top).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
