# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmrazor.models.builder import LOSSES
from mmdet.models.losses import weighted_loss
import torch

from mmcv.cnn import MODELS
from mmcv.runner import BaseModule
from mmrazor.models.builder import ARCHITECTURES
from mmfewshot.detection.models.builder import build_detector

class BaseArchitecture(BaseModule):
    """Base class for architecture.

    Args:
        model (:obj:`torch.nn.Module`): Model to be slimmed, such as
            ``DETECTOR`` in MMDetection.
    """

    def __init__(self, model, **kwargs):
        super(BaseArchitecture, self).__init__(**kwargs)
        if isinstance(model,dict):
            self.model = MODELS.build(model)
        else:
            self.model = model
    def forward_dummy(self, img):
        """Used for calculating network flops."""
        assert hasattr(self.model, 'forward_dummy')
        return self.model.forward_dummy(img)

    def forward(self, img, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return self.model(img, return_loss=return_loss, **kwargs)

    def simple_test(self, img, img_metas):
        """Test without augmentation."""
        return self.model.simple_test(img, img_metas)

    def show_result(self, img, result, **kwargs):
        """Draw `result` over `img`"""
        return self.model.show_result(img, result, **kwargs)

@ARCHITECTURES.register_module()
class MMFewShotArchitecture(BaseArchitecture):
    """Architecture based on MMFewShot."""

    def __init__(self,**kwargs):
        model = build_detector(kwargs["model"])
        super(MMFewShotArchitecture, self).__init__(model)



@LOSSES.register_module()
class LogitKnowAlignLoss(nn.Module):

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
        loss_weight=1.0,
        base_class = 3
    ):
        super(LogitKnowAlignLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.base_class = base_class
        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_T = preds_T.detach()
        
        preds_T = preds_T[:,:self.base_class]
        preds_s = preds_S[:,:self.base_class]
        preds_SN = preds_S[:,self.base_class:]

        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_s / self.tau, dim=1)
        loss_base = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        loss = loss_base
        return self.loss_weight * loss

@weighted_loss
def mse_loss(pred, target):
    """Warpper of mse loss."""
    return F.mse_loss(pred, target, reduction='none')

@LOSSES.register_module()
class FeatureKnowledgeAlignLoss(nn.Module):
    def __init__(self, tau=1.0, loss_weight=1.0):
        super(FeatureKnowledgeAlignLoss, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss

@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss