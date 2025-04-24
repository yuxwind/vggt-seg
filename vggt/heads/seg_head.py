# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .dpt_head import DPTHead
from vggt.layers.mlp import Mlp, MLP_maskformer


class SegHead(nn.Module):
    """
    Seg head that uses DPT head for pixel encoder and MLP for seg classifier
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        hidden_size=384,
        dim_mask=128,
        num_classes=150,

    ):
        """
        Initialize the SegHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            mask_dim (int): Number of feature channels in the feature extractor output.
            hidden_size (int): Size of hidden layers in the pixel encoder network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.pixel_encoder = DPTHead(
            dim_in= 2*dim_in,
            patch_size=patch_size,
            features=dim_mask,
            feature_only=True,  # Only output features, no activation
            down_ratio=1,  # TODO: Reduces spatial dimensions by factor of 2 for more efficiency?
            pos_embed=False,
        )

        # seg_predictor module that predicts semantic classes: this can be MLP or Clip. 
        # Takes  and predicts coordinates and visibility
        self.mask_classification = True
        self.aux_loss = True
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(dim_in, num_classes + 1)
        # TODO: use vggt MLP instead
        #self.mask_embed = Mlp(dim_in, hidden_size, mask_dim)
        self.mask_embed = MLP_maskformer(dim_in, hidden_size, dim_mask, 3)

    def forward(self, aggregated_tokens_list, images, patch_start_idx, aggregated_seg_tokens_list):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        B, S, _, H, W = images.shape
        
        # [N_layer, B, N_query, 1024]
        aggregated_seg_tokens_list = torch.stack(aggregated_seg_tokens_list, dim=0) 
        # Follow track_head to extract features from tokens
        # feature_maps has shape (B, S, C, H, W):[1, 25, 128, 350, 518]
        mask_features = self.pixel_encoder(aggregated_tokens_list, images, patch_start_idx)
        
        # outputs_class: torch.Size([N_layer, B, N_query, N_cls+1]): [24, 1, 100, 151]
        outputs_class = self.class_embed(aggregated_seg_tokens_list)
        out = {"pred_logits": outputs_class[-1]}

        if self.aux_loss:
            # [l, bs, queries, embed]
            # mask_embed: torch.Size([24, 1, 100, 128])
            mask_embed = self.mask_embed(aggregated_seg_tokens_list)
            # mask_features: [1, 25, 128, 350, 518]
            outputs_seg_masks = torch.einsum("lbqc,bschw->lbsqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            # FIXME h_boxes takes the last one computed, keep this in mind
            # [bs, queries, embed]
            mask_embed = self.mask_embed(aggregated_seg_tokens_list[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]



class BaseMaskClassifier(nn.Module):
    """
    Semantic segmentation mask  predictor that uses MLP/Clip for seg classifier
    """

    def __init__(
        self,
        dim_in,
        num_classes
    ):
        """
        Initialize the BaseSegPredictor module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            hidden_size (int): Size of hidden layers in the pixel encoder network.
        """
        self.class_embed = nn.Linear(dim_in, num_classes + 1)

    def forward(self, ):
        return