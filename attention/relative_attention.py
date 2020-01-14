import tensorflow as tf
from .tensor_utils import shape_list
from .attention_utils import harden_attention_weights
from .common_layers import dropout_with_broadcast_dims


def _generate_relative_positions_matrix(length_q, length_k,
                                        max_relative_position,
                                        cache=False):
  """Generates matrix of relative positions between inputs."""
  if not cache:
    if length_q == length_k:
      range_vec_q = range_vec_k = tf.range(length_q)
    else:
      range_vec_k = tf.range(length_k)
      range_vec_q = range_vec_k[-length_q:]
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
  else:
    distance_mat = tf.expand_dims(tf.range(-length_k+1, 1, 1), 0)
  distance_mat_clipped = tf.clip_by_value(distance_mat, -max_relative_position,
                                          max_relative_position)
  # Shift values to be >= 0. Each integer still uniquely identifies a relative
  # position difference.
  final_mat = distance_mat_clipped + max_relative_position
  return final_mat


def _generate_relative_positions_embeddings(length_q, length_k, depth,
                                            max_relative_position, name,
                                            cache=False):
  """Generates tensor of size [1 if cache else length_q, length_k, depth]."""
  relative_positions_matrix = _generate_relative_positions_matrix(
      length_q, length_k, max_relative_position, cache=cache)
  vocab_size = max_relative_position * 2 + 1
  # Generates embedding for each relative position of dimension depth.
  # TODO: Ensure this will work convert: tf.get_variable("name", [shape]) -> tf.Variable(initialiizer, "name")
  embeddings_table = tf.Variable(tf.constant_initializer(shape=[vocab_size, depth]), name="embeddings")
  embeddings = tf.gather(embeddings_table, relative_positions_matrix)
  return embeddings


def _relative_attention_inner(x, y, z, transpose):
  """Relative position-aware dot-product attention inner calculation.
  This batches matrix multiply calculations to avoid unnecessary broadcasting.
  Args:
    x: Tensor with shape [batch_size, heads, length or 1, length or depth].
    y: Tensor with shape [batch_size, heads, length or 1, depth].
    z: Tensor with shape [length or 1, length, depth].
    transpose: Whether to transpose inner matrices of y and z. Should be true if
        last dimension of x is depth, not length.
  Returns:
    A Tensor with shape [batch_size, heads, length, length or depth].
  """
  batch_size = tf.shape(x)[0]
  heads = x.get_shape().as_list()[1]
  length = tf.shape(x)[2]

  # xy_matmul is [batch_size, heads, length or 1, length or depth]
  xy_matmul = tf.matmul(x, y, transpose_b=transpose)
  # x_t is [length or 1, batch_size, heads, length or depth]
  x_t = tf.transpose(x, [2, 0, 1, 3])
  # x_t_r is [length or 1, batch_size * heads, length or depth]
  x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
  # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
  x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
  # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
  x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
  # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
  x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
  return xy_matmul + x_tz_matmul_r_t


def py_dot_product_attention_relative(q,
                                      k,
                                      v,
                                      bias,
                                      max_relative_position,
                                      dropout_rate=0.0,
                                      image_shapes=None,
                                      name=None,
                                      make_image_summary=True,
                                      cache=False,
                                      allow_memory=False,
                                      hard_attention_k=0,
                                      gumbel_noise_weight=0.0):
  """Calculate relative position-aware dot-product self-attention.
  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.
  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer specifying the maximum distance between
        inputs that unique position embeddings should be learned for.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    cache: whether use cache mode
    allow_memory: whether to assume that recurrent memory is in use. If True,
      the length dimension of k/v/bias may be longer than the queries, and it is
      assumed that the extra memory entries precede the non-memory entries.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
  Returns:
    A Tensor.
  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))

  # This calculation only works for self attention.
  # q, k and v must therefore have the same shape, unless memory is enabled.
  if not cache and not allow_memory:
    q.get_shape().assert_is_compatible_with(k.get_shape())
    q.get_shape().assert_is_compatible_with(v.get_shape())

  # Use separate embeddings suitable for keys and values.
  depth = k.get_shape().as_list()[3]
  length_k = shape_list(k)[2]
  length_q = shape_list(q)[2] if allow_memory else length_k
  relations_keys = _generate_relative_positions_embeddings(
      length_q, length_k, depth, max_relative_position,
      "relative_positions_keys", cache=cache)
  relations_values = _generate_relative_positions_embeddings(
      length_q, length_k, depth, max_relative_position,
      "relative_positions_values", cache=cache)

  # Compute self attention considering the relative position embeddings.
  logits = _relative_attention_inner(q, k, relations_keys, True)
  if bias is not None:
    logits += bias
    
  weights = tf.nn.softmax(logits, name="attention_weights")
  
  if hard_attention_k > 0:
    weights = harden_attention_weights(weights, hard_attention_k,
                                       gumbel_noise_weight)
                                       
  weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
  
  return _relative_attention_inner(weights, v, relations_values, False)



def _absolute_position_to_relative_position_unmasked(x):
  """Helper function for dot_product_unmasked_self_attention_relative_v2.
  Rearrange an attention logits or weights Tensor.
  The dimensions of the input represent:
  [batch, heads, query_position, memory_position]
  The dimensions of the output represent:
  [batch, heads, query_position, memory_position - query_position + length - 1]
  Only works with unmasked_attention.
  Args:
    x: a Tensor with shape [batch, heads, length, length]
  Returns:
    a Tensor with shape [batch, heads, length, 2*length-1]
  """
  batch, heads, length, _ = shape_list(x)
  # padd along column
  x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, length-1]])
  x_flat = tf.reshape(x, [batch, heads, length**2 + length*(length -1)])
  # add 0's in the beginning that will skew the elements after reshape
  x_flat = tf.pad(x_flat, [[0, 0], [0, 0], [length, 0]])
  x = tf.reshape(x_flat, [batch, heads, length, 2*length])
  x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length,
                                 2*length -1])
  return x


def get_relative_embeddings_left_right(max_relative_position, length, depth,
                                       num_heads,
                                       heads_share_relative_embedding,
                                       name):
  """Instantiate or retrieve relative embeddings, sliced according to length.
  Use for unmasked case where the relative attention looks both left and right.
  Args:
    max_relative_position: an Integer for the number of entries in the relative
      embedding, which corresponds to the max relative distance that is
      considered.
    length: an Integer, specifies the length of the input sequence for which
      this relative embedding is retrieved for.
    depth: an Integer, specifies the depth for relative embeddings.
    num_heads: an Integer, specifies the number of heads.
    heads_share_relative_embedding: a Boolean specifying if the relative
      embedding is shared across heads.
    name: a string giving the name of the embedding variables.
  Returns:
    a Tensor with shape [length, depth]
  """
  initializer_stddev = depth**-0.5
  max_relative_position_unmasked = 2 * max_relative_position - 1
  if heads_share_relative_embedding:
    embedding_shape = (max_relative_position_unmasked, depth)
  else:
    embedding_shape = (num_heads, max_relative_position_unmasked, depth)
  relative_embeddings = tf.Variable(
      tf.random.normal(stddev=initializer_stddev, shape=embedding_shape), name=name)
  # Pad first before slice to avoid using tf.cond.
  pad_length = tf.maximum(length - max_relative_position, 0)
  slice_start_position = tf.maximum(max_relative_position-length, 0)
  if heads_share_relative_embedding:
    padded_relative_embeddings = tf.pad(
        relative_embeddings,
        [[pad_length, pad_length], [0, 0]])
    used_relative_embeddings = tf.slice(
        padded_relative_embeddings,
        [slice_start_position, 0], [2 * length - 1, -1])
  else:
    padded_relative_embeddings = tf.pad(
        relative_embeddings,
        [[0, 0], [pad_length, pad_length], [0, 0]])
    used_relative_embeddings = tf.slice(
        padded_relative_embeddings,
        [0, slice_start_position, 0], [-1, 2 * length - 1, -1])
  return used_relative_embeddings


def matmul_with_relative_values(x, y, heads_share_relative_embedding):
  if heads_share_relative_embedding:
    ret = tf.einsum("bhlm,md->bhld", x, y)
  else:
    ret = tf.einsum("bhlm,hmd->bhld", x, y)
  return ret


def matmul_with_relative_keys(x, y, heads_share_relative_embedding):
  if heads_share_relative_embedding:
    ret = tf.einsum("bhld,md->bhlm", x, y)
  else:
    ret = tf.einsum("bhld,hmd->bhlm", x, y)
  return ret


def _relative_position_to_absolute_position_unmasked(x):
  """Converts tensor from relative to aboslute indexing for local attention.
  Args:
    x: a Tensor of shape [batch (or batch*num_blocks), heads,
                          length, 2 * length - 1]
  Returns:
    A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
  """
  x_shape = shape_list(x)
  batch = x_shape[0]
  heads = x_shape[1]
  length = x_shape[2]
  # Concat columns of pad to shift from relative to absolute indexing.
  col_pad = tf.zeros((batch, heads, length, 1))
  x = tf.concat([x, col_pad], axis=3)

  # Concat extra elements so to add up to shape (len+1, 2*len-1).
  flat_x = tf.reshape(x, [batch, heads, length * 2 * length])
  flat_pad = tf.zeros((batch, heads, length-1))
  flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)

  # Reshape and slice out the padded elements.
  final_x = tf.reshape(flat_x_padded, [batch, heads, length+1, 2*length-1])
  final_x = final_x[:, :, :, length-1:]
  final_x = final_x[:, :, :length, :]
  return final_x


def py_dot_product_unmasked_self_attention_relative_v2(
    q, k, v, bias, max_relative_position=None, dropout_rate=0.0,
    image_shapes=None, save_weights_to=None, name=None, make_image_summary=True,
    dropout_broadcast_dims=None, heads_share_relative_embedding=False,
    add_relative_to_values=False):
  """Calculate relative position-aware dot-product self-attention.
  The attention calculation is augmented with learned representations for the
  relative position between each element in q and each element in k and v.
  Args:
    q: a Tensor with shape [batch, heads, length, depth].
    k: a Tensor with shape [batch, heads, length, depth].
    v: a Tensor with shape [batch, heads, length, depth].
    bias: bias Tensor.
    max_relative_position: an integer the max relative embedding considered.
      Changing this invalidates checkpoints.
    dropout_rate: a floating point number.
    image_shapes: optional tuple of integer scalars.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    name: an optional string.
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    heads_share_relative_embedding: a boolean indicating wheather to share
      relative embeddings between attention heads.
    add_relative_to_values: a boolean for whether to add relative component to
      values.
  Returns:
    A Tensor.
  Raises:
    ValueError: if max_relative_position is not > 0.
  """
  if not max_relative_position:
    raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))

  # This calculation only works for self attention.
  # q, k and v must therefore have the same shape.
  q.get_shape().assert_is_compatible_with(k.get_shape())
  q.get_shape().assert_is_compatible_with(v.get_shape())

  # [batch, num_heads, query_length, memory_length]
  logits = tf.matmul(q, k, transpose_b=True)

  length = shape_list(q)[2]
  k_shape = shape_list(k)
  num_heads = k_shape[1]
  depth_k = k_shape[-1]

  key_relative_embeddings = get_relative_embeddings_left_right(
      max_relative_position, length, depth_k, num_heads,
      heads_share_relative_embedding,
      "key_relative_embeddings")
  unmasked_rel_logits = matmul_with_relative_keys(
      q, key_relative_embeddings, heads_share_relative_embedding)
  unmasked_rel_logits = _relative_position_to_absolute_position_unmasked(
      unmasked_rel_logits)
  logits += unmasked_rel_logits

  if bias is not None:
    logits += bias
  weights = tf.nn.softmax(logits, name="attention_weights")

  # dropping out the attention links for each of the heads
  weights = dropout_with_broadcast_dims(
      weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
  # relative_weights.set_shape([None, None, None, max_length])
  ret = tf.matmul(weights, v)

  if add_relative_to_values:
    # Adds the contribution of the weighted relative embeddings to the values.
    # [batch, num_heads, query_length, 2*memory_length-1]
    relative_weights = _absolute_position_to_relative_position_unmasked(
        weights)
    depth_v = shape_list(v)[3]
    value_relative_embeddings = get_relative_embeddings_left_right(
        max_relative_position, length, depth_v, num_heads,
        heads_share_relative_embedding, "value_relative_embeddings")
    ret += matmul_with_relative_values(
        relative_weights, value_relative_embeddings,
        heads_share_relative_embedding)

  return ret
