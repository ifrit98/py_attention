import tensorflow as tf
from .attention_utils import harden_attention_weights, combine_heads
from .tensor_utils import cast_like
from .common_layers import dropout_with_broadcast_dims


def py_dot_product_attention(q,
                             k,
                             v,
                             bias=None,
                             dropout_rate=0.0,
                             name=None,
                             dropout_broadcast_dims=None,
                             activation_dtype=None,
                             weight_dtype=None,
                             join_heads=False,
                             hard_attention_k=0,
                             gumbel_noise_weight=0.0):
  """Dot-product attention.
  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    name: an optional string
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    activation_dtype: Used to define function activation dtype when using
      mixed precision.
    weight_dtype: The dtype weights are stored in when using mixed precision
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
  if bias is not None:
    bias = cast_like(bias, logits)
    logits += bias
  # In case logits are fp16, upcast before softmax
  logits = tf.cast(logits, dtype=tf.float32)
  weights = tf.nn.softmax(logits, name="attention_weights")
  if hard_attention_k > 0:
    weights = harden_attention_weights(weights, hard_attention_k,
                                       gumbel_noise_weight)
  weights = cast_like(weights, q)

  # Drop out attention links for each head.
  weights = dropout_with_broadcast_dims(
      weights, dropout_rate, broadcast_dims=dropout_broadcast_dims)
  
  attention = tf.matmul(weights, v)
  
  return combine_heads(attention) if join_heads else attention

