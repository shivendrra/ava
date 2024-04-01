""" swin transformer """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange

device = 'cpu'

class Embedding(nn.Module):
    def __init__(self,patch_size=4,C=96):
        super().__init__()
        self.linear_embed = nn.Conv2d(3,C,kernel_size=patch_size,stride=patch_size)
        self.ln = nn.LayerNorm(C)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.ln(rearrange(self.linear_embed(x),'b c h w -> b (h w) c')))
        return x

class MergePatch(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.linear = nn.Linear(4*C, 2*C)
        self.ln = nn.LayerNorm(2*C)
    def forward(self,x):
        h = w = int(math.sqrt(x.shape[1])/2)
        x = rearrange(x,'b (h s1 w s2) c -> b (h w) (s1 s2 c)',s1=2,s2=2,h=h,w=w)
        return self.ln(self.linear(x))

class RelativePositionalEmbedding(nn.Module):
    def __init__(self,window_size):
        super().__init__()
        self.B = nn.Parameter(torch.zeros(2 * window_size - 1, 2 * window_size - 1))
        idx = torch.arange(window_size)
        idx = torch.stack([torch.meshgrid(idx,idx)[0].flatten(),torch.meshgrid(idx,idx)[1].flatten()])
        idx = idx[:,None,:] - idx[:,:,None]
        self.embeddings = nn.Parameter((self.B[idx[0,:,:],idx[1,:,:]]),requires_grad=False)
    def forward(self,x):
        return x+self.embeddings

class ShiftedWindowAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,window_size=7,attn_dropout=0.2,ffd_dropout=0.2,mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask = mask
        self.window_size = window_size
        self.linear = nn.Linear(embed_dim,3 * embed_dim)
        self.ffd = nn.Linear(embed_dim,embed_dim)
        self.relative_pos_embed = RelativePositionalEmbedding(window_size=window_size)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffd_dropout = nn.Dropout(ffd_dropout)
        
    def forward(self,x):
        head_dim = self.embed_dim // self.num_heads
        h = w = int(math.sqrt(x.shape[1]))
        x = self.linear(x)
        x = rearrange(x,'b (h w) (c k) -> b h w c k',k=3,h=h,w=w)
        
        if self.mask:
            x = torch.roll(x,(-self.window_size//2,-self.window_size//2),dims=(1,2))
        
        x = rearrange(x,'b (h m1) (w m2) (nh he) k -> b nh h w (m1 m2) he k',nh=self.num_heads,he=head_dim,m1=self.window_size,m2=self.window_size)
        
        Q,K,V = x.chunk(3,dim=6)
        Q,K,V = map(lambda x : x.squeeze(-1),[Q,K,V])
        w = (Q @ K.transpose(4,5))/math.sqrt(head_dim)
        w = self.relative_pos_embed(w)
        if self.mask:
            row_mask = torch.zeros((self.window_size**2,self.window_size**2)).to(device)
            row_mask[-self.window_size * (self.window_size//2):,:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[:-self.window_size * (self.window_size//2),-self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask,'(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).to(device)
            w[:,:,-1,:] += row_mask
            w[:,:,:,-1] += column_mask
        
        attention = F.softmax(w,dim=-1) @ V
        attention = self.attn_dropout(attention)
        x = rearrange(attention,'b nh h w (m1 m2) he -> b (h m1) (w m2) (nh he)',m1=self.window_size,m2=self.window_size)
        
        if self.mask:
            x = torch.roll(x,(self.window_size//2,self.window_size//2),dims=(1,2))
        
        x = rearrange(x,'b h w c -> b (h w) c')
        
        return self.ffd_dropout(self.ffd(x))

class SwinBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,mask,window_size=7):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.WMSA = ShiftedWindowAttention(embed_dim=embed_dim,num_heads=num_heads,mask=mask,window_size=window_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim,embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4,embed_dim),
        )
    def forward(self,x):
        wmsa = self.dropout(self.WMSA(self.ln(x)) + x)
        x = self.dropout(self.mlp(self.ln(wmsa)) + wmsa)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,window_size=7):
        super().__init__()
        self.wmsa = SwinBlock(embed_dim,num_heads,mask=False)
        self.swmsa = SwinBlock(embed_dim,num_heads,mask=True)
    def forward(self,x):
        return self.swmsa(self.wmsa(x))

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(patch_size=4,C=96)
        self.stage1 = nn.Sequential(
            SwinTransformerBlock(embed_dim=96,num_heads=3),
        )
        self.stage2 = nn.Sequential(
            MergePatch(96),
            SwinTransformerBlock(192,6),
        )
        self.stage3 = nn.Sequential(
            MergePatch(192),
            SwinTransformerBlock(384,12),
            SwinTransformerBlock(384,12),
            SwinTransformerBlock(384,12),
        )
        self.stage4 = nn.Sequential(
            MergePatch(384),
            SwinTransformerBlock(768,24),
        )
    def forward(self,x):
        x = self.embedding(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

model_ = SwinTransformer()
sum([p.numel() for p in model_.parameters()])