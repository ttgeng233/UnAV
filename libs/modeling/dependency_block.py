import torch
from torch import nn
from .models import register_dependency_block
from .blocks import (TransformerBlock, MaskedConv1D, LayerNorm)

@register_dependency_block("DependencyBlock")
class Dependency_Block(nn.Module):
    """
        model co-occur and temporal dependency between events in a video.
    """
    def __init__(
        self,
        in_channel,      
        n_embd,
        n_embd_ks,
        num_classes,
        path_pdrop,     
        n_head = 1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)

        self.feature_expand = MaskedConv1D(in_channel, n_embd*self.num_classes,
                        n_embd_ks, stride=1, padding=n_embd_ks//2, bias=False)

        self.cooccur_branch = TransformerBlock(n_embd, n_head, n_hidden=n_embd, path_pdrop=path_pdrop) 
        self.temporal_branch = TransformerBlock(n_embd, n_head, n_hidden=n_embd, path_pdrop=path_pdrop)

        self.feature_squeeze = MaskedConv1D(n_embd*self.num_classes, in_channel, n_embd_ks, 
                            stride=1, padding=n_embd_ks//2, bias=False)

         # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        out_features_list = tuple()
        for idx, (features, mask) in enumerate(zip(fpn_feats, fpn_masks)):
            features_exp, mask = self.feature_expand(features, mask)
            features_exp = self.relu(features_exp).view(features.shape[0], self.num_classes, -1, features.shape[-1]).contiguous()
            B, C, H, T = features_exp.size()
            #temporal denpendency 
            temp_feat = features_exp.view(-1, H, T) #[B*C, H, T]
            temp_mask = mask.repeat(C, 1, 1) #[B*C, 1, T]
            temp_output, _ = self.temporal_branch(temp_feat, temp_feat, temp_mask)
            temp_output = temp_output.view(B, C, H, T).contiguous()

            # co-occurence dependency
            coo_feat = features_exp.transpose(1,3).contiguous().view(-1, H, C) #[B*T, H, C]
            coo_mask = mask.flatten() # [B*T]
            coo_output, _ = self.cooccur_branch(coo_feat, coo_feat, coo_mask) 
            coo_output = coo_output.view(B, T, H, C).contiguous()

            output = temp_output + coo_output.transpose(1,3).contiguous()  

            out_features = output.view(output.shape[0], -1, output.shape[-1]) #[B, C*H, T]
            out_features, mask = self.feature_squeeze(out_features, mask) 

            out_features_list += (out_features,)

        return out_features_list, fpn_masks 
        
        
