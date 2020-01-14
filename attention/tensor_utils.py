import tensorflow as tf


def log_prob_from_logits(logits, reduce_axis=-1):
  return logits - tf.reduce_logsumexp(logits, axis=reduce_axis, keepdims=True)
  


def sparse_expand_dims(tensor, current_num_dims, axis=0):
  if axis == -1:
    axis = current_num_dims

  new_col = tf.zeros([tf.shape(tensor.indices)[0]], dtype=tf.int64)
  cols = tf.unstack(tensor.indices, axis=1, num=current_num_dims)
  shape = tf.unstack(tensor.dense_shape, num=current_num_dims)
  new_indices = tf.stack(cols[:axis] + [new_col] + cols[axis:], axis=1)
  return tf.SparseTensor(
      indices=new_indices,
      values=tensor.values,
      dense_shape=tf.stack(shape[:axis] + [1] + shape[axis:]))



def sparse_eye(size):
  indices = tf.cast(tf.stack([tf.range(size), tf.range(size)]), tf.int64)
  values = tf.ones(size)
  dense_shape = [tf.cast(size, tf.int64), tf.cast(size, tf.int64)]

  return tf.SparseTensor(
      indices=indices, values=values, dense_shape=dense_shape)



def to_float(x):
  """Cast x to float; created because tf.to_float is deprecated."""
  return tf.cast(x, tf.float32)



def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.compat.v1.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x



def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret



def list_product(els):
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod



def reshape_like_all_dims(a, b):
  """Reshapes a to match the shape of b."""
  ret = tf.reshape(a, tf.shape(b))
  if not tf.executing_eagerly():
    ret.set_shape(b.get_shape())
  return ret
  


def reshape_like(a, b):
  """Reshapes a to match the shape of b in all but the last dimension."""
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not tf.executing_eagerly():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret



def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [int(n), int(m // n)])



def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  Args:
    x: a Tensor with shape [..., a, b]
  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])



def combine_first_two_dimensions(x):
  """Reshape x so that the first two dimension become one.
  Args:
    x: a Tensor with shape [a, b, ...]
  Returns:
    a Tensor with shape [ab, ...]
  """
  ret = tf.reshape(x, tf.concat([[-1], shape_list(x)[2:]], 0))
  old_shape = x.get_shape().dims
  a, b = old_shape[:2]
  new_shape = [a * b if a and b else None] + old_shape[2:]
  ret.set_shape(new_shape)
  return ret


