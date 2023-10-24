import torch
from torch import nn
from torch.nn import functional as F

from .models import register_multimodal_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, 
                    MaskedConv1D, LayerNorm, TemporalMaxer)


@register_multimodal_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in_V,                # input visual feature dimension
        n_in_A,                # input audio feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        scale_factor = 2,      # dowsampling rate for the branch,
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        branch_type = "unAV",
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.branch_type = branch_type

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]-1): 
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        #cross-attention on original temporal resolution
        self.ori_cross_att_Va = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_cross_att_Av = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        #cross-attention after down-sampling
        self.cross_att_Va = nn.ModuleList()
        self.cross_att_Av = nn.ModuleList()
        if branch_type == "maxer": 
            for idx in range(arch[2]):
                self.cross_att_Va.append(TemporalMaxer(kernel_size=3, stride=self.scale_factor, padding=1, n_embd=n_embd)) 
                self.cross_att_Av.append(TemporalMaxer(kernel_size=3, stride=self.scale_factor, padding=1, n_embd=n_embd))
        else:
            
            for idx in range(arch[2]):
                self.cross_att_Va.append(TransformerBlock(
                        n_embd, n_head,
                        n_ds_strides=(self.scale_factor, self.scale_factor),
                        attn_pdrop=attn_pdrop,
                        proj_pdrop=proj_pdrop,
                        path_pdrop=path_pdrop,
                    )
                )
                self.cross_att_Av.append(TransformerBlock(
                        n_embd, n_head,
                        n_ds_strides=(self.scale_factor, self.scale_factor),
                        attn_pdrop=attn_pdrop,
                        proj_pdrop=proj_pdrop,
                        path_pdrop=path_pdrop,
                    )
                )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        # x_V/x_A: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C_V, T = x_V.size()
        mask_V = mask_A = mask
        # embedding network
        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        # stem transformer
        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

        x_Va, mask_V = self.ori_cross_att_Va(x_V, x_A, mask_V) 
        x_Av, mask_V = self.ori_cross_att_Av(x_A, x_V, mask_A) 

        # prep for outputs
        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks_V = tuple()
        out_masks_A = tuple()
        # 1x resolution
        out_feats_V += (x_Va, )
        out_masks_V += (mask_V, )
        out_feats_A += (x_Av, )
        out_masks_A += (mask_A, )

        # main branch with downsampling
        for idx in range(len(self.cross_att_Va)):
            if self.branch_type == "maxer": 
                x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], mask_V) 
                x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], mask_A)
            else:
                x_V, mask_V = self.cross_att_Va[idx](out_feats_V[idx], out_feats_A[idx], mask_V) 
                x_A, mask_A = self.cross_att_Av[idx](out_feats_A[idx], out_feats_V[idx], mask_A)

            out_feats_V += (x_V, )
            out_masks_V += (mask_V, )

            out_feats_A += (x_A, )
            out_masks_A += (mask_A, )

        return out_feats_V, out_feats_A, out_masks_V