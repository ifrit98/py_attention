
# from .common_layers import apply_norm, apply_spectral_norm, layer_norm, group_norm, noam_norm, l2_norm, instance_norm
# from .activations import belu, brelu, gelu, nac, nalu
from .tensor_utils import cast_like, shape_list, reshape_like
from .common_layers import conv_block_downsample, diagonal_conv_gru, conv_gru
from .common_layers import sepconv_relu_sepconv, conv, conv1d, conv_block, conv1d_block, general_conv
from .activations import lrelu, hard_sigmoid, saturating_sigmoid, hard_tanh, inverse_exp_decay, inverse_lin_decay, inverse_sigmoid_decay
from .attention_utils import py_compute_qkv, shape_list, combine_heads, split_heads

from .local_attention import py_local_attention_1d
from .dilated_attention import py_dilated_self_attention_1d, py_masked_dilated_self_attention_1d
from .multihead_attention import py_multihead_attention
from .relative_attention import py_dot_product_attention_relative, py_dot_product_unmasked_self_attention_relative_v2
from .dot_product_attention import py_dot_product_attention
from .area_attention import py_dot_product_area_attention
from .WeightNorm import WeightNormalization
