import tensorflow as tf
from .tensor_utils import shape_list, to_float
from .attention_utils import reshape_by_blocks, embedding_to_padding
from .dot_product_attention import py_dot_product_attention

from numpy import tri, ones


def gather_dilated_memory_blocks(x,
                                 num_memory_blocks,
                                 gap_size,
                                 query_block_size,
                                 memory_block_size,
                                 gather_indices,
                                 direction="left"):
  """Gathers blocks with gaps in between.
  Args:
    x: Tensor of shape [length, batch, heads, depth]
    num_memory_blocks: how many memory blocks to look in "direction". Each will
      be separated by gap_size.
    gap_size: an integer indicating the gap size
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gather_indices: The indices to gather from.
    direction: left or right
  Returns:
    Tensor of shape [batch, heads, blocks, block_length, depth]
  """
  gathered_blocks = []
  # gathering memory blocks
  for block_id in range(num_memory_blocks):
    block_end_index = -(query_block_size + gap_size *
                        (block_id + 1) + memory_block_size * block_id)
    block_start_index = (
        (memory_block_size + gap_size) * (num_memory_blocks - (block_id + 1)))
    if direction != "left":
      [block_end_index,
       block_start_index] = [-block_start_index, -block_end_index]
    if block_end_index == 0:
      x_block = x[block_start_index:]
    else:
      x_block = x[block_start_index:block_end_index]

    def gather_dilated_1d_blocks(x, gather_indices):
      x_new = tf.gather(x, gather_indices)
      # [batch, heads, blocks, block_length, dim]
      return tf.transpose(x_new, [2, 3, 0, 1, 4])

    gathered_blocks.append(gather_dilated_1d_blocks(x_block, gather_indices))
  return tf.concat(gathered_blocks, 3)


def py_dilated_self_attention_1d(q,
                              k,
                              v,
                              query_block_size=128,
                              memory_block_size=128,
                              gap_size=2,
                              num_memory_blocks=2,
                              name=None):
  """Dilated self-attention.
  Args:
    q: a Tensor with shape [batch, heads, length, depth]
    k: a Tensor with shape [batch, heads, length, depth]
    v: a Tensor with shape [batch, heads, length, depth]
    query_block_size: an integer indicating size of query block
    memory_block_size: an integer indicating the size of a memory block.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left and right.
      Each will be separated by gap_size.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth]
  """
  v_list_shape = v.get_shape().as_list()
  assert v_list_shape == k.shape.as_list(), "K and V depths must be equal"
  v_shape = shape_list(v)
  depth_v = v_shape[3]
  batch_size = v_shape[0]
  num_heads = v_shape[1]
  original_length = shape_list(q)[2]

  # Pad query, key, value to ensure multiple of corresponding lengths.
  def pad_to_multiple(x, pad_length):
    x_length = shape_list(x)[2]
    return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

  def pad_l_and_r(x, pad_length):
    return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

  q = pad_to_multiple(q, query_block_size)
  v = pad_to_multiple(v, query_block_size)
  k = pad_to_multiple(k, query_block_size)

  # Set up query blocks.
  new_q_shape = shape_list(q)
  q = reshape_by_blocks(q, new_q_shape, query_block_size)
  self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
  self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)

  # Set up key and value windows.
  k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
  k = pad_l_and_r(k, k_v_padding)
  v = pad_l_and_r(v, k_v_padding)

  # Get gather indices.
  index_length = (new_q_shape[2] - query_block_size + memory_block_size)
  indices = tf.range(0, index_length, delta=1, name="index_range")
  indices = tf.reshape(indices, [1, -1, 1])  # [1, length, 1] for convs
  kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
  gather_indices = tf.nn.conv1d(
      tf.cast(indices, tf.float32),
      kernel,
      query_block_size,
      padding="VALID",
      name="gather_conv")

  gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

  # Get left and right memory blocks for each query.
  # [length, batch, heads, dim]
  k_t = tf.transpose(k, [2, 0, 1, 3])
  v_t = tf.transpose(v, [2, 0, 1, 3])
  left_k = gather_dilated_memory_blocks(
      k_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
      query_block_size, memory_block_size, gather_indices)
  left_v = gather_dilated_memory_blocks(
      v_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
      query_block_size, memory_block_size, gather_indices)

  right_k = gather_dilated_memory_blocks(
      k_t[k_v_padding:, :, :, :],
      num_memory_blocks,
      gap_size,
      query_block_size,
      memory_block_size,
      gather_indices,
      direction="right")
  right_v = gather_dilated_memory_blocks(
      v_t[k_v_padding:, :, :, :],
      num_memory_blocks,
      gap_size,
      query_block_size,
      memory_block_size,
      gather_indices,
      direction="right")

  k_windows = tf.concat([left_k, self_k_part, right_k], axis=3)
  v_windows = tf.concat([left_v, self_v_part, right_v], axis=3)
  attention_bias = tf.expand_dims(
      to_float(embedding_to_padding(k_windows)) * -1e9, axis=-2)

  output = py_dot_product_attention(
      q,
      k_windows,
      v_windows,
      attention_bias,
      dropout_rate=0.,
      name="dilated_1d")
  output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

  # Remove the padding if introduced.
  output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
  output.set_shape(v_list_shape)
  return output


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.
  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.
  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    # TODO: rework to use tensorflow ops?
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = tri(cols, rows, num_lower).T
    upper_mask = tri(rows, cols, num_upper)
    band = ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.linalg.band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band


def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.
  A position may attend to positions at most max_distance from it,
  forward and backwards.
  This does not actually save any computation.
  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.
  Allows a query to attend to all positions up to and including its own.
  Args:
   length: a Scalar.
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  return attention_bias_local(length, -1, 0)


def py_masked_dilated_self_attention_1d(q,
                                        k,
                                        v,
                                        query_block_size=64,
                                        memory_block_size=64,
                                        gap_size=2,
                                        num_memory_blocks=2,
                                        name=None):
  """Dilated self-attention. TODO(avaswani): Try it and write a paper on it.
  Args:
    q: a Tensor with shape [batch, heads, length, depth]
    k: a Tensor with shape [batch, heads, length, depth]
    v: a Tensor with shape [batch, heads, length, depth]
    query_block_size: an integer
    memory_block_size: an integer indicating how much to look left.
    gap_size: an integer indicating the gap size
    num_memory_blocks: how many memory blocks to look at to the left. Each will
      be separated by gap_size.
    name: an optional string
  Returns:
    a Tensor of shape [batch, heads, length, depth]
  """
  v_list_shape = v.get_shape().as_list()
  assert v_list_shape == k.shape.as_list(), "K and V depths must be equal"
  v_shape = shape_list(v)
  depth_v = v_shape[3]
  batch_size = v_shape[0]
  num_heads = v_shape[1]
  original_length = shape_list(q)[2]

  # Pad query, key, value to ensure multiple of corresponding lengths.
  def pad_to_multiple(x, pad_length):
    x_length = shape_list(x)[2]
    return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

  def pad_l(x, left_pad_length):
    return tf.pad(x, [[0, 0], [0, 0], [left_pad_length, 0], [0, 0]])

  q = pad_to_multiple(q, query_block_size)
  v = pad_to_multiple(v, query_block_size)
  k = pad_to_multiple(k, query_block_size)

  # Set up query blocks.
  new_q_shape = shape_list(q)
  q = reshape_by_blocks(q, new_q_shape, query_block_size)

  # Set up key and value windows.
  self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
  self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)
  k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
  k = pad_l(k, k_v_padding)
  v = pad_l(v, k_v_padding)

  # Get gather indices.
  index_length = (new_q_shape[2] - query_block_size + memory_block_size)

  indices = tf.range(0, index_length, delta=1, name="index_range")
  indices = tf.reshape(indices, [1, -1, 1])  # [1, length, 1] for convs
  kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
  gather_indices = tf.nn.conv1d(
      tf.cast(indices, tf.float32),
      kernel,
      query_block_size,
      padding="VALID",
      name="gather_conv")
  gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

  # Get left and right memory blocks for each query.
  # [length, batch, heads, dim]
  k_t = tf.transpose(k, [2, 0, 1, 3])
  v_t = tf.transpose(v, [2, 0, 1, 3])

  k_unmasked_windows = gather_dilated_memory_blocks(
      k_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
      gather_indices)
  v_unmasked_windows = gather_dilated_memory_blocks(
      v_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
      gather_indices)

  # Combine memory windows.
  block_q_shape = shape_list(q)
  masked_attention_bias = tf.tile(
      tf.expand_dims(attention_bias_lower_triangle(query_block_size), axis=0),
      [block_q_shape[0], block_q_shape[1], block_q_shape[2], 1, 1])
  padding_attention_bias = tf.expand_dims(
      embedding_to_padding(k_unmasked_windows) * -1e9, axis=-2)
  padding_attention_bias = tf.tile(padding_attention_bias,
                                    [1, 1, 1, query_block_size, 1])
  attention_bias = tf.concat(
      [masked_attention_bias, padding_attention_bias], axis=-1)
  # combine memory windows
  k_windows = tf.concat([self_k_part, k_unmasked_windows], 3)
  v_windows = tf.concat([self_v_part, v_unmasked_windows], 3)
  output = py_dot_product_attention(
      q,
      k_windows,
      v_windows,
      attention_bias,
      dropout_rate=0.,
      name="dilated_1d")
  output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

  # Remove the padding if introduced.
  output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
  output.set_shape(v_list_shape)
  return output
