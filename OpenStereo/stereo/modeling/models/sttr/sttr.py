#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.
import torch
import torch.nn as nn

from .utilities.feat_extractor_backbone_in import build_backbone
from .utilities.feat_extractor_tokenizer import build_tokenizer
from .utilities.pos_encoder import build_position_encoding
from .utilities.regression_head import build_regression_head
from .utilities.transformer import build_transformer
from .utilities.misc import batched_index_select, NestedTensor
from .utilities.loss import build_criterion
from .utilities import Map


class STTR(nn.Module):
    """
    STTR: it consists of
        - backbone: contracting path of feature descriptor
        - tokenizer: expanding path of feature descriptor
        - pos_encoder: generates relative sine pos encoding
        - transformer: computes self and cross attention
        - regression_head: regresses disparity and occlusion, including optimal transport
    """

    def __init__(self, args):
        super(STTR, self).__init__()
        layer_channel = [64, 128, 128]

        self.downsample = args.DOWNSAMPLE

        self.backbone = build_backbone(args)
        self.tokenizer = build_tokenizer(args, layer_channel)
        self.pos_encoder = build_position_encoding(args)
        self.transformer = build_transformer(args)
        self.regression_head = build_regression_head(args)

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()

        loss_cfg = {'px_error_threshold': 3, 'validation_max_disp': 192, 'loss_weight': 'rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0'}
        loss_cfg = Map(loss_cfg)
        self.criterion = build_criterion(loss_cfg)

    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                # fix the bug mentioned in https://github.com/mli0603/stereo-transformer/issues/8
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.inplace = True

    def forward(self, inputs):
        """
        :param x: input data
        :return:
            a dictionary object with keys
            - "disp_pred" [N,H,W]: predicted disparity
            - "occ_pred" [N,H,W]: predicted occlusion mask
            - "disp_pred_low_res" [N,H//s,W//s]: predicted low res (raw) disparity
        """
        left, right = inputs['left'], inputs['right']
        occ_mask = inputs['occ_mask'].bool()
        occ_mask_right = inputs['occ_mask_right'].bool()
        disp = inputs['disp']

        device = left.get_device()
        downsample = self.downsample
        bs, _, h, w = left.size()
        if downsample <= 0:
            sampled_cols = None
            sampled_rows = None
        else:
            col_offset = int(downsample / 2)
            row_offset = int(downsample / 2)
            sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).to(device)
            sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).to(device)

        x = NestedTensor(left, right, sampled_cols=sampled_cols, sampled_rows=sampled_rows, disp=disp,
                                     occ_mask=occ_mask, occ_mask_right=occ_mask_right)
        bs, _, h, w = x.left.size()

        # extract features
        feat = self.backbone(x)  # concatenate left and right along the dim=0
        tokens = self.tokenizer(feat)  # 2NxCxHxW
        pos_enc = self.pos_encoder(x)  # NxCxHx2W-1

        # separate left and right
        feat_left = tokens[:bs]
        feat_right = tokens[bs:]  # NxCxHxW

        # downsample
        if x.sampled_cols is not None:
            feat_left = batched_index_select(feat_left, 3, x.sampled_cols)
            feat_right = batched_index_select(feat_right, 3, x.sampled_cols)
        if x.sampled_rows is not None:
            feat_left = batched_index_select(feat_left, 2, x.sampled_rows)
            feat_right = batched_index_select(feat_right, 2, x.sampled_rows)

        # transformer
        attn_weight = self.transformer(feat_left, feat_right, pos_enc)

        # regress disparity and occlusion
        output = self.regression_head(attn_weight, x)

        output['input_nested'] = x
        return output

    def get_loss(self, model_preds, input_data):
        inputs = model_preds.pop('input_nested')
        losses = self.criterion(inputs, model_preds)
        total_loss = losses['aggregated']

        loss_info = {'scalar/train/loss_disp': total_loss.item()}
        return total_loss, loss_info
