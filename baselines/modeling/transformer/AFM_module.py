import torch
import torch.nn as nn
import torch.nn.functional as F


class AFM(nn.Module):
    """
    Attention-Aware Filtering Module (AFM)
    
    This module performs three-step filtering on cost maps using CLIP attention weights:
    1. Attention-based Patch Filtering
    2. Class Confidence-based Filtering  
    3. Attention-aware Cost Refinement
    
    Args:
        num_layers (int): Number of CLIP attention layers (default: 12)
        K (float): Threshold multiplier for attention filtering (default: 1.0)
        tau (float): Threshold for class confidence filtering (default: 0.1)
    """
    
    def __init__(self, num_layers=12, K=6, tau=0.2):
        super(AFM, self).__init__()
        self.num_layers = num_layers
        self.K = K  # 阈值倍数
        self.tau = tau  # 置信度阈值
        
        # 无需训练参数
        
    def attention_based_patch_filtering(self, attention_matrices_list):
        """
        Attention-based Patch Filtering
        
        Args:
            attention_matrices_list: List of multi-layer self-attention matrices,
                                   each layer is [B, N, N] where N is number of patches
        
        Returns:
            M_attn: Attention-based patch pair mask [B, N, N]
        """
        attention_matrices = torch.stack(attention_matrices_list, dim=1)  # [B, L, N, N]
        B, L, N, _ = attention_matrices.shape

        A_bar = torch.mean(attention_matrices, dim=(2, 3), keepdim=True)  # [B, L, 1, 1]

        layer_masks = (attention_matrices > A_bar).float()  # [B, L, N, N]

        vote_count = torch.sum(layer_masks, dim=1)  # [B, N, N]

        M_attn = (vote_count > self.K).float()  # [B, N, N]
        
        return M_attn
    
    def class_confidence_based_filtering(self, P_coarse):
        """
        Class Confidence-based Filtering
        
        Args:
            P_coarse: Initial classification probabilities [B, N, C]
                     where N is number of patches, C is number of classes
        
        Returns:
            M_cls: Class confidence mask [B, N, C]
        """
        B, N, C = P_coarse.shape
        
        max_prob, _ = torch.max(P_coarse, dim=2, keepdim=True)  # [B, N, 1]
        relative_confidence = P_coarse - max_prob  # [B, N, C]

        M_cls = (relative_confidence > -self.tau).float()  # [B, N, C]
        
        return M_cls
    
    def attention_aware_cost_refinement(self, attention_matrices_list, M_attn, M_cls, P_coarse):
        """
        Attention-aware Cost Refinement
        
        Args:
            attention_matrices_list: List of multi-layer attention matrices,
                                   each layer is [B, N, N]
            M_attn: Attention mask [B, N, N]
            M_cls: Class confidence mask [B, N, C]
            P_coarse: Initial probabilities [B, N, C]
        
        Returns:
            P_refined: Refined probability distribution [B, N, C]
        """

        attention_matrices = torch.stack(attention_matrices_list, dim=1)  # [B, L, N, N]
        B, L, N, _ = attention_matrices.shape
        _, _, C = P_coarse.shape

        M_attn_expanded = M_attn.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, N, N]

        masked_attention = attention_matrices * M_attn_expanded  # [B, L, N, N]

        avg_attention = torch.mean(masked_attention, dim=1)  # [B, N, N]

        masked_cost = P_coarse * M_cls  # [B, N, C]

        P_refined = torch.bmm(avg_attention, masked_cost)  # [B, N, C]
        
        return P_refined
    
    def forward(self, attention_matrices_list, P_coarse):
        """
        Forward pass of AFM module
        
        Args:
            attention_matrices_list: List of CLIP attention weights from 12 layers,
                                   each layer is [B, N, N] where N is number of patches (H*W)
            P_coarse: Initial cost/probability map [B, N, C]
                     where C is number of classes
        
        Returns:
            P_refined: Refined probability distribution [B, N, C]
        """

        M_attn = self.attention_based_patch_filtering(attention_matrices_list)

        M_cls = self.class_confidence_based_filtering(P_coarse)

        P_refined = self.attention_aware_cost_refinement(
            attention_matrices_list, M_attn, M_cls, P_coarse
        )
        
        return P_refined