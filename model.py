import torch
from torch import nn
import numpy  as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18,resnet34,resnet50,resnet101
from torchvision.models._utils import IntermediateLayerGetter

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        #dys# q (sz_b,n_head，N=len_q,d_k)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #dys# attn= (sz_b,n_head，N=len_q,d_k)*(sz_b,n_head,d_k,len_q)=(sz_b,n_head，N=len_q,N=len_q)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)##dys# output(sz_b,n_head，N=len_q,d_k)

        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) #dys#(sz_b,N=len_q,n_head，d_k)->(sz_b,n_head，N=len_q,d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        #q (sz_b,n_head,N=len_q,d_k)
        #k (sz_b,n_head,N=len_k,d_k)
        #v (sz_b,n_head,N=len_v,d_v)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        #q (sz_b,len_q,n_head,N * d_k)##
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #dys# x(sz_b, len_q, d_model)
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None): # dys# enc_input (sz_b,len_q,d_model)
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)  # dys# enc_output :(sz_b,len_q,d_model), enc_slf_attn:(sz_b,n_head，N=len_q,N=len_q)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self,  n_layers, n_head, d_k, d_v,d_model, d_inner,  dropout=0.1):
        super(Encoder,self).__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)for _ in range(n_layers)])

    def forward(self,enc_input, src_mask=None, return_attns=False):
        enc_slf_attn_list = []
        enc_output=enc_input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class BackBone_net(nn.Module):
    def __init__(self,modelname,d_model,pretrained=False,backbone_feature_dim=None,unit_channel=3):
        super(BackBone_net, self).__init__()
        self.backbone=None
        self.unit_channel=unit_channel
        print("*"*100,modelname)
        if modelname in ["resnet18","resnet34"]:
            backbone_feature_dim=512*1
        elif modelname in ["resnet50","resnet101"]:
            backbone_feature_dim = 512 * 4

        assert backbone_feature_dim is not  None,print("The feature dimension of the backbone needs to be specified.")
        if modelname =="resnet18":
            self.backbone=resnet18(pretrained=pretrained)
        elif modelname =="resnet34":
            self.backbone=resnet34(pretrained=pretrained)
        elif modelname =="resnet50":
            self.backbone=resnet50(pretrained=pretrained)
        elif modelname =="resnet101":
            self.backbone=resnet101(pretrained=pretrained)
        assert self.backbone is not None, print("Need to specify the backbone.")
        if (unit_channel!=3 and  unit_channel!=1): ##
            self.backbone.conv1 = nn.Conv2d(unit_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone=IntermediateLayerGetter(self.backbone,return_layers={"layer4":"featuremaps"})##
        self.conv_out= nn.Conv2d(backbone_feature_dim,d_model,  kernel_size=1,stride=1)
        self.backbone_feature_dim=d_model
        self.w_dim_div=32
        self.h_dim_div=32
    def forward(self,x):
        if self.unit_channel== 1:
            x=repeat(x,"b c h w -> b (c r ) h w",r=3)##
        x=self.backbone(x)["featuremaps"]# dys#feature map
        x=self.conv_out(x)
        return x

class Transformer_2d(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self,backbone_name,image_size, num_classes,total_channels=15,unit_channel= 3, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,pretrained=True,backbone_feature_dim=None,pool = 'cls',feature_patch_d=1,feature_patch_h=1):
        super(Transformer_2d,self).__init__()
        total_channels=total_channels*3
        self.backbone=BackBone_net(modelname=backbone_name,d_model=d_model,pretrained=pretrained,backbone_feature_dim=backbone_feature_dim,unit_channel=unit_channel)
        self.w_dim_div=self.backbone.w_dim_div
        self.h_dim_div=self.backbone.h_dim_div
        # self.backbone_feature_dim=self.backbone.backbone_feature_dim
        image_height, image_width = pair(image_size)
        assert image_height % self.w_dim_div== 0 and image_width % self.h_dim_div == 0 and (total_channels% unit_channel== 0  ) , \
           print('Image dimensions must be divisible by the patch size',image_height ,self.w_dim_div, image_width ,total_channels)
        self.pool=pool
        channel_tie_0 = total_channels // unit_channel
        channel_tie=channel_tie_0
        num_patches = (image_height // self.h_dim_div) * (image_width // self.w_dim_div) * (
                    channel_tie ) // feature_patch_h // feature_patch_h // feature_patch_d  ##
        self.backbone_feature_dim = self.backbone.backbone_feature_dim * feature_patch_h * feature_patch_h * feature_patch_d

        print("feature_patch_d,feature_patch_h):", feature_patch_d, feature_patch_h)
        self.img_to_patch = nn.Sequential(Rearrange('b c d h w  -> (b d ) c h w '))
        self.feaure_to_patch_0 = Rearrange('(b c1) c h w -> b c1 c h w  ', c1=channel_tie_0)
        self.feaure_to_patch=nn.Sequential(Rearrange('(b d1) c ( h h1) ( w w1) -> b (c h1 w1) d1 h w  ',d1=channel_tie,h1= feature_patch_h,w1=feature_patch_h),
                                           Rearrange(' b c (d d1 ) h w-> b (c d1) (d h w)  ', d1=feature_patch_d) )

        self.to_patch_embedding = nn.Linear(self.backbone_feature_dim,
                                            d_model) if self.backbone_feature_dim != d_model else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_enc = PositionalEncoding(d_model, n_position=num_patches + 1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.encoder = Encoder(d_model=d_model, d_inner=d_inner,
                               n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.classifier = nn.Sequential( nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes - 1, bias=False))

    def forward(self, img):

        x = self.img_to_patch(img)
        x = self.backbone(x)
        x = self.feaure_to_patch(x)
        x = x.transpose(-1, -2)
        x = self.to_patch_embedding(x)
        #### -- Forward
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(self.position_enc(x))
        x = self.layer_norm(x)
        ##  encoder
        x = self.encoder(x)
        ## output
        x = x[:, 0] if self.pool == "cls" else (x.mean(dim=1) if self.pool == "mean" else x.max(dim=1)[0])
        x = self.classifier(x)
        cls = nn.Sigmoid()(x)
        return cls

