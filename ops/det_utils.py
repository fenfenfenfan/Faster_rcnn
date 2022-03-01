import imp
from nis import match
from operator import concat
from turtle import width
from numpy import dtype
import torch
import math
from typing import List, Tuple
from torch import Tensor


class BoxCoder(object):

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # per_image_anchors_num
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_num = 0
        for val in boxes_per_image:
            box_sum += val

        pred_boxes = self.decode_single(rel_codes, concat_boxes)

        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        # xmin,ymin,xmax,ymax
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 0]
        ctr_x = boxes[:, 0] + widths * 0.5
        ctr_y = boxes[:, 1] + heights * 0.5

        wx, wy, ww, wh = self.weights

        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # proposal xmin,ymin,xmax,ymax
        pred_boxes1 = pred_ctr_x - torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4),
            dim=2).flatten(1)
        return pred_boxes

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + ex_widths * 0.5
    ex_ctr_y = proposals_y1 + ex_heights * 0.5

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + gt_widths * 0.5
    gt_ctr_y = reference_boxes_y1 + gt_heights * 0.5

    target_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    target_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    target_dw = ww * torch.log(gt_widths / ex_widths)
    target_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((target_dx, target_dy, target_dw, target_dh), dim=1)
    return targets


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLD = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLD': int,
    }

    def __init__(self,
                 high_threshold,
                 low_threshold,
                 allow_low_quaility_matches=False):
        # type: (float, float, bool) -> None
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLD = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quaility_matches = allow_low_quaility_matches

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quaility_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        below_low_threshold = matched_vals < self.low_threshold
        between_threshold = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_threshold] = self.BETWEEN_THRESHOLD

        if self.allow_low_quaility_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches,
                                          match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches,
                                 match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:,
                                                                      None]))

        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class BalancedPositiveNegativeSampler(object):

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            negative = torch.where(torch.ge(matched_idxs_per_image, 0))[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.nueml(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.nueml(),
                                   device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.nueml(),
                                   device=positive.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                                                      dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image,
                                                      dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def smooth_l1_loss(input,
                   target,
                   beta: float = 1. / 9,
                   size_average: bool = True):
    n = torch.abs(input - target)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum