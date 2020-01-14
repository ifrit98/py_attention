

def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position
  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(1, 1+length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position
  Returns:
    a Tensor the same shape as x.
  """
  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                start_index)
  return x + common_layers.cast_like(signal, x)


def get_layer_timing_signal_learned_1d(channels, layer, num_layers):
  """get n-dimensional embedding as the layer (vertical) timing signal.
  Adds embeddings to represent the position of the layer in the tower.
  Args:
    channels: dimension of the timing signal
    layer: layer num
    num_layers: total number of layers
  Returns:
    a Tensor of timing signals [1, 1, channels].
  """
  shape = [num_layers, 1, 1, channels]
  layer_embedding = (
      tf.get_variable(
          "layer_embedding",
          shape,
          initializer=tf.random_normal_initializer(0, channels**-0.5)) *
      (channels**0.5))
  return layer_embedding[layer, :, :, :]


def add_layer_timing_signal_learned_1d(x, layer, num_layers):
  """Add n-dimensional embedding as the layer (vertical) timing signal.
  Adds embeddings to represent the position of the layer in the tower.
  Args:
    x: a tensor with shape [batch, length, depth]
    layer: layer num
    num_layers: total number of layers
  Returns:
    a Tensor the same shape as x.
  """
  channels = common_layers.shape_list(x)[-1]
  signal = get_layer_timing_signal_learned_1d(channels, layer, num_layers)
  x += signal
  return x


def get_layer_timing_signal_sinusoid_1d(channels, layer, num_layers):
  """Add sinusoids of different frequencies as layer (vertical) timing signal.
  Args:
    channels: dimension of the timing signal
    layer: layer num
    num_layers: total number of layers
  Returns:
    a Tensor of timing signals [1, 1, channels].
  """

  signal = get_timing_signal_1d(num_layers, channels)
  layer_signal = tf.expand_dims(signal[:, layer, :], axis=1)

  return layer_signal


def add_layer_timing_signal_sinusoid_1d(x, layer, num_layers):
  """Add sinusoids of different frequencies as layer (vertical) timing signal.
  Args:
    x: a Tensor with shape [batch, length, channels]
    layer: layer num
    num_layers: total number of layers
  Returns:
    a Tensor the same shape as x.
  """

  channels = common_layers.shape_list(x)[-1]
  signal = get_layer_timing_signal_sinusoid_1d(channels, layer, num_layers)

  return x + signal


def add_timing_signal_1d_given_position(x,
                                        position,
                                        min_timescale=1.0,
                                        max_timescale=1.0e4):
  """Adds sinusoids of diff frequencies to a Tensor, with timing position given.
  Args:
    x: a Tensor with shape [batch, length, channels]
    position: a Tensor with shape [batch, length]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  channels = common_layers.shape_list(x)[2]
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = (
      tf.expand_dims(tf.to_float(position), 2) * tf.expand_dims(
          tf.expand_dims(inv_timescales, 0), 0))
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
  signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
  signal = common_layers.cast_like(signal, x)
  return x + signal


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  expressed in terms of b, sin(a) and cos(a).
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
  num_dims = len(x.get_shape().as_list()) - 2
  channels = common_layers.shape_list(x)[-1]
  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  for dim in range(num_dims):
    length = common_layers.shape_list(x)[dim + 1]
    position = tf.to_float(tf.range(length))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    prepad = dim * 2 * num_timescales
    postpad = channels - (dim + 1) * 2 * num_timescales
    signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
    for _ in range(1 + dim):
      signal = tf.expand_dims(signal, 0)
    for _ in range(num_dims - 1 - dim):
      signal = tf.expand_dims(signal, -2)
    x += signal
  return x


def add_positional_embedding(x, max_length, name=None, positions=None):
  """Adds positional embedding.
  Args:
    x: Tensor with shape [batch, length, depth].
    max_length: int representing static maximum size of any dimension.
    name: str representing name of the embedding tf.Variable.
    positions: Tensor with shape [batch, length].
  Returns:
    Tensor of same shape as x.
  """
  with tf.name_scope("add_positional_embedding"):
    _, length, depth = common_layers.shape_list(x)
    var = tf.cast(tf.get_variable(name, [max_length, depth]), x.dtype)
    if positions is None:
      pad_length = tf.maximum(0, length - max_length)
      sliced = tf.cond(
          tf.less(length, max_length),
          lambda: tf.slice(var, [0, 0], [length, -1]),
          lambda: tf.pad(var, [[0, pad_length], [0, 0]]))
      return x + tf.expand_dims(sliced, 0)
    else:
      return x + tf.gather(var, tf.to_int32(positions))


def add_positional_embedding_nd(x, max_length, name=None):
  """Adds n-dimensional positional embedding.
  The embeddings add to all positional dimensions of the tensor.
  Args:
    x: Tensor with shape [batch, p1 ... pn, depth]. It has n positional
      dimensions, i.e., 1 for text, 2 for images, 3 for video, etc.
    max_length: int representing static maximum size of any dimension.
    name: str representing name of the embedding tf.Variable.
  Returns:
    Tensor of same shape as x.
  """
  with tf.name_scope("add_positional_embedding_nd"):
    x_shape = common_layers.shape_list(x)
    num_dims = len(x_shape) - 2
    depth = x_shape[-1]
    base_shape = [1] * (num_dims + 1) + [depth]
    base_start = [0] * (num_dims + 2)
    base_size = [-1] + [1] * num_dims + [depth]
    for i in range(num_dims):
      shape = base_shape[:]
      start = base_start[:]
      size = base_size[:]
      shape[i + 1] = max_length
      size[i + 1] = x_shape[i + 1]
      var = tf.get_variable(
          name + "_%d" % i,
          shape,
          initializer=tf.random_normal_initializer(0, depth**-0.5))
      var = var * depth**0.5
      x += tf.slice(var, start, size)
    return x
