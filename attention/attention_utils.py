import tensorflow as tf
from .tensor_utils import split_last_dimension, combine_last_two_dimensions, shape_list, to_float
from .common_layers import dense, conv1d


def py_compute_attention_component(antecedent,
                                   total_depth,
                                   filter_width=1,
                                   padding="VALID",
                                   name="c",
                                   vars_3d_num_heads=0,
                                   layer_collection=None):
  """Computes attention component (query, key or value).
  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    c : [batch, length, depth] tensor
  """
  if layer_collection is not None:
    if filter_width != 1 or vars_3d_num_heads != 0:
      raise ValueError(
          "KFAC implementation only supports filter_width=1 (actual: {}) and "
          "vars_3d_num_heads=0 (actual: {}).".format(
              filter_width, vars_3d_num_heads))
  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.Variable(
      tf.random.normal([input_depth,
      vars_3d_num_heads,
      total_depth // vars_3d_num_heads],
      stddev=initializer_stddev), name = name)
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return dense(
        antecedent, total_depth, use_bias=False, name=name,
        layer_collection=layer_collection)
  else:
    return conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)


def py_compute_qkv(query_antecedent,
                   memory_antecedent,
                   total_key_depth,
                   total_value_depth,
                   q_filter_width=1,
                   kv_filter_width=1,
                   q_padding="VALID",
                   kv_padding="VALID",
                   vars_3d_num_heads=0,
                   layer_collection=None):
  """Computes query, key and value.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = py_compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  k = py_compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  v = py_compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  return q, k, v



def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])



def split_heads_2d(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, height, width, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, height, width, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_heads_nd(x, num_heads):
  """Split the depth dimension (last dimension) into multiple heads.
  Args:
    x: a [batch, d1, ..., dn, depth] tensor
    num_heads: an integer
  Returns:
    a [batch, num_heads, d1, ..., dn, depth // num_heads]
  """
  num_dimensions = len(shape_list(x)) - 2
  return tf.transpose(
      split_last_dimension(x, num_heads), [0, num_dimensions + 1] +
      list(range(1, num_dimensions + 1)) + [num_dimensions + 2])



def combine_heads(x):
  """Inverse of split_heads.
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))



def combine_heads_2d(x):
  """Inverse of split_heads_2d.
  Args:
    x: a Tensor with shape
      [batch, num_heads, height, width, channels / num_heads]
  Returns:
    a Tensor with shape [batch, height, width, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads_nd(x):
  """Inverse of split_heads_nd.
  Args:
    x: a [batch, num_heads, d1, ..., dn, depth // num_heads] tensor
  Returns:
    a [batch, d1, ...., dn, depth] tensor
  """
  num_dimensions = len(shape_list(x)) - 3
  return combine_last_two_dimensions(
      tf.transpose(x, [0] + list(range(2, num_dimensions + 2)) +
                   [1, num_dimensions + 2]))



def reshape_by_blocks(x, x_shape, memory_block_size):
  """Reshapes input by splitting its length over blocks of memory_block_size.
  Args:
    x: a Tensor with shape [batch, heads, length, depth]
    x_shape: tf.TensorShape of x.
    memory_block_size: Integer which divides length.
  Returns:
    Tensor with shape
    [batch, heads, length // memory_block_size, memory_block_size, depth].
  """
  x = tf.reshape(x, [
      x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
      memory_block_size, x_shape[3]
  ])
  return x
  
  
  
def top_kth_iterative(x, k):
  """Compute the k-th top element of x on the last axis iteratively.
  This assumes values in x are non-negative, rescale if needed.
  It is often faster than tf.nn.top_k for small k, especially if k < 30.
  Note: this does not support back-propagation, it stops gradients!
  Args:
    x: a Tensor of non-negative numbers of type float.
    k: a python integer.
  Returns:
    a float tensor of the same shape as x but with 1 on the last axis
    that contains the k-th largest number in x.
  """
  # The iterative computation is as follows:
  #
  # cur_x = x
  # for _ in range(k):
  #   top_x = maximum of elements of cur_x on the last axis
  #   cur_x = cur_x where cur_x < top_x and 0 everywhere else (top elements)
  #
  # We encode this computation in a TF graph using tf.foldl, so the inner
  # part of the above loop is called "next_x" and tf.foldl does the loop.
  def next_x(cur_x, _):
    top_x = tf.reduce_max(cur_x, axis=-1, keep_dims=True)
    return cur_x * to_float(cur_x < top_x)
  # We only do k-1 steps of the loop and compute the final max separately.
  fin_x = tf.foldl(next_x, tf.range(k - 1), initializer=tf.stop_gradient(x),
                   parallel_iterations=2, back_prop=False)
  return tf.stop_gradient(tf.reduce_max(fin_x, axis=-1, keep_dims=True))



def harden_attention_weights(weights, k, gumbel_noise_weight):
  """Make attention weights non-0 only on the top k ones."""
  if gumbel_noise_weight > 0.:
    gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(weights),
                                                     minval=1e-5,
                                                     maxval=1 - 1e-5)))
    weights += gumbel_noise * gumbel_noise_weight

  # Subtract the top-kth weight and zero-out all lower ones.
  # Note that currently in case of numerical ties it will retain more
  # than k elements. In the future, we may want to avoid this.
  weights -= top_kth_iterative(weights, k)
  weights = tf.nn.relu(weights)
  # Re-normalize the weights.
  weights_sum = tf.reduce_sum(weights, axis=-1, keep_dims=True)
  weights_sum = tf.maximum(weights_sum, 1e-6)  # Avoid division by 0.
  weights /= weights_sum
  return weights



def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.
  We have hacked symbol_modality to return all-zero embeddings for padding.
  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
    embedding vector is all zero, and is 0 otherwise.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.cast(tf.equal(emb_sum, 0.0), tf.int32)



def padding_to_length(padding):
  """Calculate the length of mask based on padding.
  Args:
    padding: a Tensor with shape [..., length].
  Returns:
    a Tensor with shape [...].
  """
  non_padding = 1.0 - padding
  return tf.cast(tf.reduce_sum(non_padding, axis=-1), tf.int32)
