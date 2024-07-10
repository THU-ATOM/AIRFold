# -*- coding: utf-8 -*-
'''
@Time    : 2022/4/28 19:29
@Author  : ld
@File    : interface.py
'''
from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class ModelInput():
    """
    Input type of for model.
    """

    sequences: List[Union[torch.LongTensor, str]]
    model_coords: Optional[torch.FloatTensor] = None
    model_mask: Optional[torch.BoolTensor] = None
    coords_label: Optional[torch.FloatTensor] = None
    return_embeddings: Optional[bool] = False


@dataclass
class ModelOutput():
    """
    Output type of for model.
    """
    coords: torch.FloatTensor
    p_lddt_pred: torch.FloatTensor
    translations: torch.FloatTensor
    rotations: torch.FloatTensor
    coords_loss: Optional[torch.FloatTensor] = None
    torsion_loss: Optional[torch.FloatTensor] = None
    bondlen_loss: Optional[torch.FloatTensor] = None
    p_lddt_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


