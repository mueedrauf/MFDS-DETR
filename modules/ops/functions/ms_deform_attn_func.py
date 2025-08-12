# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F

# Removed import MultiScaleDeformableAttention as MSDA and MSDeformAttnFunction class
# This file now only provides the pure PyTorch implementation for CPU-only usage

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Multi-scale deformable attention, samples features from multi-scale value tensors
    according to sampling locations, then applies attention weights and sums.
    This is a pure PyTorch implementation for CPU-only usage.
    """
    # N_: batch, S_: len_in, M_: head, D_: d_model // head
    N_, S_, M_, D_ = value.shape
    # Lq_: len_query, M_: head, L_: num_feature_map, P_: num_points
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # Split value tensor into list for each feature level
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # Normalize sampling locations to [-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        # Fix shape: grid_sample returns (N_*M_, D_, Lq_, P_)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    # torch.stack(sampling_value_list, dim=-2).flatten(-2): (N_*M_, D_, Lq_, L_, P_) -> (N_*M_, D_, Lq_, L_*P_)
    stacked = torch.stack(sampling_value_list, dim=-2)  # (N_*M_, D_, Lq_, L_, P_)
    stacked = stacked.flatten(-2)  # (N_*M_, D_, Lq_, L_*P_)
    output = (stacked * attention_weights).sum(-1)  # (N_*M_, D_, Lq_)
    output = output.view(N_, Lq_, M_ * D_)  # (N, Lq, M*D)
    return output