import tensorflow as tf

from .tensor_utils import shape_list
from .attention_utils import reshape_by_blocks, combine_heads, embedding_to_padding

from .dot_product_attention import py_dot_product_attention


def py_local_attention_1d(q, k, v, block_length=128, filter_width=100, 
                          join_heads=True, name=None):
  """Strided block local self-attention.
  The sequence is divided into blocks of length block_length. Attention for a
  given query position can see all memory positions in the corresponding block
  and filter_width many positions to the left and right of the block.
  Args:
    q: a Tensor with shape [batch, heads, length, depth_k]
    k: a Tensor with shape [batch, heads, length, depth_k]
    v: a Tensor with shape [batch, heads, length, depth_v]
    block_length: an integer
    filter_width: an integer indicating how much to look left and right of the
      block.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth_v]
  """
  # Check that q, k, v have the same shape except in their depth dimension.
  q.get_shape()[:-1].assert_is_compatible_with(k.get_shape()[:-1])
  q.get_shape()[:-1].assert_is_compatible_with(v.get_shape()[:-1])

  # TODO: replace shape_list() with q.shape.as_list() ?
  batch_size, num_heads, original_length, _ = shape_list(q)

  # Pad query, key, value to ensure multiple of corresponding lengths.
  def pad_to_multiple(x, pad_length):
    x_length = shape_list(x)[2]
    return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

  def pad_l_and_r(x, pad_length):
    return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

  # Set up query blocks.
  # [batch, heads, blocks_q, block_length, depth_k]
  q = pad_to_multiple(q, block_length)
  q = reshape_by_blocks(q, shape_list(q), block_length)
  total_query_blocks = shape_list(q)[2]

  # Set up key and value blocks.
  # [batch, heads, blocks_k, block_length, depth_k]
  blocks_per_filter_width = filter_width // block_length
  remaining_items = filter_width % block_length
  k = pad_to_multiple(k, block_length)
  v = pad_to_multiple(v, block_length)
  k = pad_l_and_r(k, filter_width + block_length - remaining_items)
  v = pad_l_and_r(v, filter_width + block_length - remaining_items)
  k = reshape_by_blocks(k, shape_list(k), block_length)
  v = reshape_by_blocks(v, shape_list(v), block_length)

  total_kv_blocks = shape_list(k)[2]

  slices = []
  # prepare the left-most and right-most partial blocks if needed
  if remaining_items:
    first_partial_block_k = tf.slice(
        k, [0, 0, 0, block_length - remaining_items, 0],
        [-1, -1, total_query_blocks, -1, -1])
    first_partial_block_v = tf.slice(
        v, [0, 0, 0, block_length - remaining_items, 0],
        [-1, -1, total_query_blocks, -1, -1])
    last_partial_block_k = tf.slice(
        k, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
        [-1, -1, -1, remaining_items, -1])
    last_partial_block_v = tf.slice(
        v, [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
        [-1, -1, -1, remaining_items, -1])
    slices.append((first_partial_block_k, first_partial_block_v))
    slices.append((last_partial_block_k, last_partial_block_v))

  # Prepare the rest of the blocks
  first_block_index = 1 if remaining_items else 0
  attention_blocks = 2 * blocks_per_filter_width + 1
  for i in range(first_block_index, attention_blocks + first_block_index):
    block_k = tf.slice(k, [0, 0, i, 0, 0],
                       [-1, -1, total_query_blocks, -1, -1])
    block_v = tf.slice(v, [0, 0, i, 0, 0],
                       [-1, -1, total_query_blocks, -1, -1])
    slices.append((block_k, block_v))
  # [batch, heads, blocks_q, block_length + 2 * filter_width, depth_k]
  k = tf.concat([s[0] for s in slices], axis=3)
  v = tf.concat([s[1] for s in slices], axis=3)

  print(-1e9)
  pad = tf.cast(embedding_to_padding(k), tf.float32)
  print(pad)
  attention_bias = tf.expand_dims(pad * tf.cast(-1e9, tf.float32), axis=-2)
  depth_v = shape_list(v)[-1]

  output = py_dot_product_attention(
      q,
      k,
      v,
      attention_bias,
      dropout_rate=0.,
      name="py_local_1d")
  output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

  # Remove the padding if introduced.
  output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
  output.set_shape([None if isinstance(dim, tf.Tensor) else dim for dim in
                    (batch_size, num_heads, original_length, depth_v)])

  if join_heads:
    output = combine_heads(output)
                    
  return output
