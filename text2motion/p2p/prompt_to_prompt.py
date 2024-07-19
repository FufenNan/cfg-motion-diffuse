from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
import p2p.ptp_utils
import p2p.seq_aligner
import math
from torch.nn import functional as F
LOW_RESOURCE = False
    
def register_attention_control(encoder, controller):
    def ca_forward(self, place_in_unet):
        # to_out = self.to_out
        # if type(to_out) is torch.nn.modules.container.ModuleList:
        #     to_out = self.to_out[0]
        # else:
        #     to_out = self.to_out
        def forward(x, xf, emb):
            if len(x.shape)==len(xf.shape):
                is_cross = True
                """
                x: B, T, D
                xf: B, N, L
                """
                B, T, D = x.shape
                N = xf.shape[1]
                H = self.num_head
                # B, T, 1, D
                query = self.query(self.norm(x)).unsqueeze(2)
                # B, 1, N, D
                key = self.key(self.text_norm(xf)).unsqueeze(1)
                query = query.view(B, T, H, -1)
                key = key.view(B, N, H, -1)
                # B, T, N, H
                attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
                weight = self.dropout(F.softmax(attention, dim=2))
                weight = controller(weight, is_cross, place_in_unet)
                value = self.value(self.text_norm(xf)).view(B, N, H, -1)
                y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
                y = x + self.proj_out(y, emb)
                return y
            elif len(x.shape)!=len(xf.shape):
                src_mask=emb
                emb_new=xf
                is_cross = False
                """
                x: B, T, D
                """
                B, T, D = x.shape
                H = self.num_head
                # B, T, 1, D
                query = self.query(self.norm(x)).unsqueeze(2)
                # B, 1, T, D
                key = self.key(self.norm(x)).unsqueeze(1)
                query = query.view(B, T, H, -1)
                key = key.view(B, T, H, -1)
                # B, T, T, H
                attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
                attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000
                weight = self.dropout(F.softmax(attention, dim=2))
                weight = controller(weight, is_cross, place_in_unet)
                value = self.value(self.norm(x)).view(B, T, H, -1)
                y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
                y = x + self.proj_out(y, emb_new)
                return y
        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        #对Crossattention做修改
        if 'CrossAttention' in net_.__class__.__name__:
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        # elif hasattr(net_, 'children'):
        #     for net__ in net_.children():
        #         count = register_recr(net__, count, place_in_unet)
        return count
    #遍历model中的每个模块
    cross_att_count = 0
    for index, module in enumerate(encoder.temporal_decoder_blocks):
        sub_nets=module.named_children()
        for j,block in enumerate(sub_nets):
            cross_att_count+=register_recr(block[1],0,f"layer{index}")
    controller.num_att_layers = cross_att_count