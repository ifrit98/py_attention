import tensorflow as tf
from .tensor_utils import to_float, shape_list, cast_like
from .activations import hard_sigmoid, saturating_sigmoid



def layers():
  """Get the layers module good for TF 1 and TF 2 work for now."""
  # layers_module = None
  # try:
  #   layers_module = tf.layers
  # except AttributeError:
  #   logging.info("Cannot access tf.layers, trying TF2 layers.")
  # try:
  #   from tensorflow.python import tf2  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
  #   if tf2.enabled():
  #     logging.info("Running in V2 mode, using Keras layers.")
  #     layers_module = tf.keras.layers
  # except ImportError:
  #   pass
  # _cached_layers = layers_module
  
  layers_module = tf.keras.layers # Assumes tf>=2.0
  return layers_module


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
  """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.
  Instead of specifying noise_shape, this function takes broadcast_dims -
  a list of dimension numbers in which noise_shape should be 1.  The random
  keep/drop tensor has dimensionality 1 along these dimensions.
  Args:
    x: a floating point tensor.
    keep_prob: A scalar Tensor with the same type as x.
      The probability that each element is kept.
    broadcast_dims: an optional list of integers
      the dimensions along which to broadcast the keep/drop flags.
    **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
  Returns:
    Tensor of the same shape as x.
  """
  assert "noise_shape" not in kwargs
  if broadcast_dims is not None:
    shape = tf.shape(x)
    ndims = len(x.get_shape())
    # Allow dimensions like "-1" as well.
    broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
    kwargs["noise_shape"] = [
        1 if i in broadcast_dims else shape[i] for i in range(ndims)
    ]
  return tf.nn.dropout(x, keep_prob, **kwargs)


def comma_separated_string_to_integer_list(s):
  return [int(i) for i in s.split(",") if i]


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.Variable(tf.ones(shape=[filters]), name="layer_norm_scale")
  bias  = tf.Variable(tf.zeros(shape=[filters]), name="layer_norm_bias")
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias):
  """Layer norm raw computation."""

  # Save these before they get converted to tensors by the casting below
  params = (scale, bias)

  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(
      tf.math.squared_difference(x, mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)

  output = norm_x * scale + bias


  return output


def layer_norm(x,
               filters=None,
               epsilon=None,
               name=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if epsilon is None:
    epsilon = 1e-6
  if filters is None:
    filters = shape_list(x)[-1]
  scale, bias = layer_norm_vars(filters)
  return layer_norm_compute(x, epsilon, scale, bias)


def group_norm(x, filters=None, num_groups=8, epsilon=None):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""
  x_shape = shape_list(x)
  if epsilon is None:
    epsilon = 1e-5
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4
  assert filters % num_groups == 0
  # Prepare variables.
  scale = tf.Variable(tf.ones(shape=[filters]), name="group_norm_scale")
  bias = tf.Variable(tf.zeros(shape=[filters]), name="group_norm_scale")
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # Reshape and compute group norm.
  x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
  # Calculate mean and variance on heights, width, channels (not groups).
  mean, variance = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
  norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
  return tf.reshape(norm_x, x_shape) * scale + bias


def noam_norm(x, epsilon):
  """One version of layer normalization."""
  if epsilon is None:
    epsilon = 1.0
  with tf.name_scope("noam_norm"):
    shape = x.get_shape()
    ndims = len(shape)
    return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
        to_float(shape[-1])))


def l2_norm(x, filters=None, epsilon=None, name=None, reuse=None):
  """Layer normalization with l2 norm."""
  if epsilon is None:
    epsilon = 1e-6
  if filters is None:
    filters = shape_list(x)[-1]

  scale = tf.Variable(tf.ones(shape=[filters]), name="l2_norm_scale")
  bias = tf.Variable(tf.zeros(shape=[filters]), name="l2_norm_bias")
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  l2norm = tf.reduce_sum(
      tf.math.squared_difference(x, mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.math.rsqrt(l2norm + epsilon)
  return norm_x * scale + bias


def apply_spectral_norm(x):
  """Normalizes x using the spectral norm.
  The implementation follows Algorithm 1 of
  https://arxiv.org/abs/1802.05957. If x is not a 2-D Tensor, then it is
  reshaped such that the number of channels (last-dimension) is the same.
  Args:
    x: Tensor with the last dimension equal to the number of filters.
  Returns:
    x: Tensor with the same shape as x normalized by the spectral norm.
    assign_op: Op to be run after every step to update the vector "u".
  """
  weights_shape = shape_list(x)
  other, num_filters = tf.reduce_prod(weights_shape[:-1]), weights_shape[-1]

  # Reshape into a 2-D matrix with outer size num_filters.
  weights_2d = tf.reshape(x, (other, num_filters))

  # v = Wu / ||W u||
  u = tf.Variable(tf.compat.v1.truncated_normal(shape=[num_filters, 1]), 
                  name="u", trainable = False)
  v = tf.nn.l2_normalize(tf.matmul(weights_2d, u))

  # u_new = vW / ||v W||
  u_new = tf.nn.l2_normalize(tf.matmul(tf.transpose(v), weights_2d))

  # s = v*W*u
  spectral_norm = tf.squeeze(
      tf.matmul(tf.transpose(v), tf.matmul(weights_2d, tf.transpose(u_new))))

  # set u equal to u_new in the next iteration.
  assign_op = tf.compat.v1.assign(u, tf.transpose(u_new))
  return tf.divide(x, spectral_norm), assign_op



def instance_norm(x):
  """Instance normalization layer."""
  epsilon = 1e-5
  mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
  scale = tf.Variable(
    tf.compat.v1.truncated_normal(shape=[x.get_shape()[-1]], mean=1.0, stddev=0.02),
    name="scale"
  )
  offset = tf.Variable(
    tf.zeros(shape=[x.get_shape()[-1]]),
    name="offset"
  )
  # offset = tf.get_variable(
  #     "offset", [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))

  out = scale * tf.math.divide(x - mean, tf.sqrt(var + epsilon)) + offset

  return out


def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "instance":
    return instance_norm(x)
  if norm_type == "layer":
    return layer_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "batch":
    return layers().BatchNormalization(epsilon=epsilon)(x)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "l2":
    return l2_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none" or not norm_type:
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")


def zero_add(previous_value, x, name=None, reuse=None):
  """Resnet connection with zero initialization.
  Another type of resnet connection which returns previous_value + gamma * x.
  gamma is a trainable scalar and initialized with zero. It is useful when a
  module is plugged into a trained model and we want to make sure it matches the
  original model's performance.
  Args:
    previous_value:  A tensor.
    x: A tensor.
    name: name of variable scope; defaults to zero_add.
    reuse: reuse scope.
  Returns:
    previous_value + gamma * x.
  """
  # TODO make sure this works as intented...
  gamma = tf.Variable(tf.zeros(shape=()), name="gamma")
  return previous_value + gamma * x



def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         name=None,
                         dropout_broadcast_dims=None):
  """Apply a sequence of functions to the input or output of a layer.
  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout
    z: zero add
  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))
  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    norm_type: a string (see apply_norm())
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    a Tensor
  """
  if sequence is None:
    return x
  for c in sequence:
    if c == "a":
      x += previous_value
    elif c == "z":
      x = zero_add(previous_value, x)
    elif c == "n":
      x = apply_norm(
          x, norm_type, depth, epsilon)
    else:
      assert c == "d", ("Unknown sequence step %s" % c)
      x = dropout_with_broadcast_dims(
          x, dropout_rate, broadcast_dims=dropout_broadcast_dims)
  return x


def layer_preprocess(layer_input, norm_type, norm_epsilon, sequence, dropout, broadcast_dims=None):
  """Apply layer preprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon
  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    a Tensor
  """
  assert "a" not in sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  assert "z" not in sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=sequence,
      dropout_rate=dropout,
      norm_type=norm_type,
      depth=None,
      epsilon=norm_epsilon,
      dropout_broadcast_dims=broadcast_dims)


def layer_postprocess(layer_input, layer_output, sequence, norm_type, 
                      norm_epsilon, dropout, broadcast_dims=None):
  """Apply layer postprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon
  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.
  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=sequence,
      dropout_rate=dropout,
      norm_type=norm_type,
      depth=None,
      epsilon=norm_epsilon,
      dropout_broadcast_dims=broadcast_dims)


def general_conv(x,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding="VALID",
                 name="conv",
                 do_norm="instance",
                 do_relu=True,
                 relufactor=0):
  """Generalized convolution layer."""
  x = layers().Conv2D(
      num_filters,
      filter_size,
      stride,
      padding,
      activation=None,
      kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
      bias_initializer=tf.constant_initializer(0.0))(x)
  if do_norm == "layer":
    x = layer_norm(x)
  elif do_norm == "instance":
    x = instance_norm(x)

  if do_relu:
    if relufactor == 0:
      x = tf.nn.relu(x, "relu")
    else:
      x = lrelu(x, leak=relufactor)

  return x


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if kwargs.get("padding") == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
    original_name = kwargs.pop("name", None)
    original_force2d = kwargs.pop("force2d", None)
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
  def _conv2d(x, *args, **kwargs):
    return layers().Conv2D(*args, **kwargs)(x)
  return conv_internal(
      _conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return tf.squeeze(
      conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1),
           dilation_rate=(dilation_rate, 1), **kwargs),
      2)


def separable_conv(inputs, filters, kernel_size, **kwargs):
  def _sep_conv2d(x, *args, **kwargs):
    return layers().SeparableConv2D(*args, **kwargs)(x)
  return conv_internal(_sep_conv2d, inputs, filters, kernel_size, **kwargs)


def subseparable_conv(inputs, filters, kernel_size, **kwargs):
  """Sub-separable convolution. If separability == 0 it's a separable_conv."""

  def conv_fn(inputs, filters, kernel_size, **kwargs):
    """Sub-separable convolution, splits into separability-many blocks."""
    separability = None
    if "separability" in kwargs:
      separability = kwargs.pop("separability")
    if separability:
      parts = []
      abs_sep = separability if separability > 0 else -1 * separability
      for split_idx, split in enumerate(tf.split(inputs, abs_sep, axis=3)):
        if separability > 0:
          parts.append(
              layers().Conv2D(filters // separability, kernel_size,
                              **kwargs)(split))
        else:
          parts.append(
              layers().SeparableConv2D(filters // abs_sep,
                                        kernel_size, **kwargs)(split))
      if separability > 1:
        result = layers().Conv2D(filters, (1, 1))(tf.concat(parts, axis=3))
      elif abs_sep == 1:  # If we have just one block, return it.
        assert len(parts) == 1
        result = parts[0]
      else:
        result = tf.concat(parts, axis=3)
    else:
      result = layers().SeparableConv2D(filters, kernel_size,
                                        **kwargs)(inputs)
    if separability is not None:
      kwargs["separability"] = separability
    return result

  return conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs)


def dense(x, units, **kwargs):
  """Identical to layers.dense."""
  layer_collection = kwargs.pop("layer_collection", None)
  activations = layers().Dense(units, **kwargs)(x)
  if layer_collection:
    # We need to find the layer parameters using scope name for the layer, so
    # check that the layer is named. Otherwise parameters for different layers
    # may get mixed up.
    layer_name = tf.compat.v1.get_variable_scope().name
    if (not layer_name) or ("name" not in kwargs):
      raise ValueError(
          "Variable scope and layer name cannot be empty. Actual: "
          "variable_scope={}, layer name={}".format(
              layer_name, kwargs.get("name", None)))

    layer_name += "/" + kwargs["name"]
    layer_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                     scope=layer_name)
    assert layer_params
    if len(layer_params) == 1:
      layer_params = layer_params[0]

    tf.compat.v1.logging.info(
        "Registering dense layer to collection for tensor: {}".format(
            layer_params))

    x_shape = x.shape.as_list()
    if len(x_shape) == 3:
      # Handle [batch, time, depth] inputs by folding batch and time into
      # one dimension: reshaping inputs to [batchxtime, depth].
      x_2d = tf.reshape(x, [-1, x_shape[2]])
      activations_shape = activations.shape.as_list()
      activations_2d = tf.reshape(activations, [-1, activations_shape[2]])
      layer_collection.register_fully_connected_multi(
          layer_params, x_2d, activations_2d, num_uses=x_shape[1])
      activations = tf.reshape(activations_2d, activations_shape)
    else:
      layer_collection.register_fully_connected(layer_params, x, activations)
  return activations



def batch_dense(inputs,
                units,
                activation=None,
                kernel_initializer=None,
                reuse=None,
                name=None):
  """Multiply a batch of input matrices by a batch of parameter matrices.
  Each input matrix is multiplied by the corresponding parameter matrix.
  This is useful in a mixture-of-experts where the batch represents different
  experts with different inputs.
  Args:
    inputs: a Tensor with shape [batch, length, input_units]
    units: an integer
    activation: an optional activation function to apply to the output
    kernel_initializer: an optional initializer
    reuse: whether to reuse the varaible scope
    name: an optional string
  Returns:
    a Tensor with shape [batch, length, units]
  Raises:
    ValueError: if the "batch" or "input_units" dimensions of inputs are not
      statically known.
  """
  inputs_shape = shape_list(inputs)
  if len(inputs_shape) != 3:
    raise ValueError("inputs must have 3 dimensions")
  batch = inputs_shape[0]
  input_units = inputs_shape[2]
  if not isinstance(batch, int) or not isinstance(input_units, int):
    raise ValueError("inputs must have static dimensions 0 and 2")

  if kernel_initializer is None:
    kernel_initializer = tf.random_normal_initializer(
        stddev=input_units**-0.5)
  w = tf.Variable(
    kernel_initializer(shape=[batch, input_units, units], dtype=inputs.dtype),
    name="w"
  )
  # w = tf.get_variable(
  #     "w", [batch, input_units, units],
  #     initializer=kernel_initializer,
  #     dtype=inputs.dtype)
  y = tf.matmul(inputs, w)
  if activation is not None:
    y = activation(y)
  return y



def conv_block_internal(conv_fn,
                        inputs,
                        filters,
                        dilation_rates_and_kernel_sizes,
                        first_relu=True,
                        use_elu=False,
                        separabilities=None,
                        **kwargs):
  """A block of convolutions.
  Args:
    conv_fn: convolution function, e.g. conv or separable_conv.
    inputs: a Tensor
    filters: an Integer
    dilation_rates_and_kernel_sizes: a list of tuples (dilation, (k_w, k_h))
    first_relu: whether to do a relu at start (defaults to True)
    use_elu: whether to use ELUs instead of ReLUs (defaults to False)
    separabilities: list of separability factors (per-layer).
    **kwargs: additional arguments (e.g., pooling)
  Returns:
     a Tensor.
  """

  name = kwargs.pop("name") if "name" in kwargs else None
  mask = kwargs.pop("mask") if "mask" in kwargs else None

  # Usage for normalize_fn kwarg:
  # if not specified, use layer norm
  # if given normalize_fn=None, don't use any normalization
  # if given normalize_fn=norm, use the specified norm function

  use_layer_norm = "normalizer_fn" not in kwargs
  norm = kwargs.pop("normalizer_fn", None)
  use_normalizer_fn = use_layer_norm or norm

  if use_layer_norm:
    norm = lambda x, name: layer_norm(x, filters, name=name)

  cur, counter = inputs, -1
  for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
    counter += 1
    if first_relu or counter > 0:
      cur = tf.nn.elu(cur) if use_elu else tf.nn.relu(cur)
    if mask is not None:
      cur *= mask
    if separabilities:
      cur = conv_fn(
          cur,
          filters,
          kernel_size,
          dilation_rate=dilation_rate,
          name="conv_block_%d" % counter,
          use_bias=norm is None,
          separability=separabilities[counter],
          **kwargs)
    else:
      cur = conv_fn(
          cur,
          filters,
          kernel_size,
          dilation_rate=dilation_rate,
          name="conv_block_%d" % counter,
          use_bias=norm is None,
          **kwargs)
    if use_normalizer_fn:
      cur = norm(cur, name="conv_block_norm_%d" % counter)
  return cur


def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 2d convolutions."""
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def conv1d_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 1d convolutions."""
  return conv_block_internal(conv1d, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def separable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                         **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(separable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def subseparable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                            **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(subseparable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def pool(inputs, window_size, pooling_type, padding, strides=(1, 1)):
  """Pooling (supports "LEFT")."""
  with tf.name_scope("pool", values=[inputs]):
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
      raise ValueError("Inputs to conv must have statically known rank 4.")
    # Add support for left padding.
    if padding == "LEFT":
      assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1
      if len(static_shape) == 3:
        width_padding = 2 * (window_size[1] // 2)
        padding_ = [[0, 0], [width_padding, 0], [0, 0]]
      else:
        height_padding = 2 * (window_size[0] // 2)
        cond_padding = tf.cond(
            tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
            lambda: tf.constant(2 * (window_size[1] // 2)))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
      inputs = tf.pad(inputs, padding_)
      inputs.set_shape([static_shape[0], None, None, static_shape[3]])
      padding = "VALID"

  return tf.nn.pool(inputs, window_size, pooling_type, padding, strides=strides)


def conv_block_downsample(x,
                          kernel,
                          strides,
                          padding,
                          separability=0,
                          name=None,
                          reuse=None):
  """Implements a downwards-striding conv block, like Xception exit flow."""
  hidden_size = int(x.get_shape()[-1])
  res = conv_block(
      x,
      int(1.25 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      strides=strides,
      name="res_conv")

  x = subseparable_conv_block(
      x,
      hidden_size, [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv0")
  x = subseparable_conv_block(
      x,
      int(1.25 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv1")
  x = pool(x, kernel, "MAX", padding, strides=strides)

  x += res

  x = subseparable_conv_block(
      x,
      2 * hidden_size, [((1, 1), kernel)],
      first_relu=False,
      padding=padding,
      separability=separability,
      name="conv2")
  x = subseparable_conv_block(
      x,
      int(2.5 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv3")
  return x


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """If necessary, zero out inputs to a conv for padding positions.
  Args:
    inputs: a Tensor with shape [batch, length, ...]
    kernel_size: an integer or pair of integers
    nonpadding_mask: a Tensor with shape [batch, length]
  Returns:
    Tensor of the same shape as inputs.
  """
  if (kernel_size != 1 and kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask

  return inputs


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   decode_loop_step=None):
  """Hidden layer with RELU activation followed by linear projection.
  Args:
    inputs: A tensor.
    filter_size: An integer.
    output_size: An integer.
    first_kernel_size: An integer.
    second_kernel_size: An integer.
    padding: A string.
    nonpadding_mask: A tensor.
    dropout: A float.
    name: A string.
    cache: A dict, containing Tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU. If it is not None, the function
        will do inplace update for the cache instead of concatenating the
        current result to the cache.
  Returns:
    A Tensor.
  """
  from tensorflow.python.ops import inplace_ops

  inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)

  if cache:
    if decode_loop_step is None:
      inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
    else:
      # Inplace update is required for inference on TPU.
      # Inplace_ops only supports inplace_update on the first dimension.
      # The performance of current implementation is better than updating
      # the tensor by adding the result of matmul(one_hot,
      # update_in_current_step)
      tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
      tmp_f = inplace_ops.alias_inplace_update(
          tmp_f,
          decode_loop_step * tf.shape(inputs)[1],
          tf.transpose(inputs, perm=[1, 0, 2]))
      inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
    inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

  h = conv1d(
      inputs, filter_size, first_kernel_size, padding=padding, name="conv1")

  if cache:
    h = h[:, -1:, :]

  h = tf.nn.relu(h)
  if dropout != 0.0:
    h = tf.nn.dropout(h, 1.0 - dropout)
  h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
  return conv1d(
      h, output_size, second_kernel_size, padding=padding, name="conv2")




def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     layer_collection=None,
                     name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  # layer_name is appended with "conv1" or "conv2" in this method only for
  # historical reasons. These are in fact dense layers.
  layer_name = "%s_{}" % name if name else "{}"
  h = dense(
      inputs,
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
      layer_collection=layer_collection,
      name=layer_name.format("conv1"))

  if dropout != 0.0:
    h = dropout_with_broadcast_dims(
        h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
  o = dense(
      h,
      output_size,
      activation=output_activation,
      use_bias=True,
      layer_collection=layer_collection,
      name=layer_name.format("conv2"))
  return o




def sepconv_relu_sepconv(inputs,
                         filter_size,
                         output_size,
                         first_kernel_size=(1, 1),
                         second_kernel_size=(1, 1),
                         padding="LEFT",
                         nonpadding_mask=None,
                         dropout=0.0,
                         name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)
  if inputs.get_shape().ndims == 3:
    is_3d = True
    inputs = tf.expand_dims(inputs, 2)
  else:
    is_3d = False
  h = separable_conv(
      inputs,
      filter_size,
      first_kernel_size,
      activation=tf.nn.relu,
      padding=padding,
      name="conv1")
  if dropout != 0.0:
    h = tf.nn.dropout(h, 1.0 - dropout)
  h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
  ret = separable_conv(
      h, output_size, second_kernel_size, padding=padding, name="conv2")
  if is_3d:
    ret = tf.squeeze(ret, 2)
  return ret


def conv_gru(x,
             kernel_size,
             filters,
             padding="SAME",
             dilation_rate=(1, 1),
             name=None,
             reuse=None):
  """Convolutional GRU in 1 dimension."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start, padding):
    return conv(
        args,
        filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  reset = saturating_sigmoid(do_conv(x, "reset", 1.0, padding))
  gate = saturating_sigmoid(do_conv(x, "gate", 1.0, padding))
  candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0, padding))
  return gate * x + (1 - gate) * candidate
    


def conv_lstm(x,
              kernel_size,
              filters,
              padding="SAME",
              dilation_rate=(1, 1),
              name=None,
              reuse=None):
  """Convolutional LSTM in 1 dimension."""
  gates = conv(
      x,
      4 * filters,
      kernel_size,
      padding=padding,
      dilation_rate=dilation_rate)
  g = tf.split(layer_norm(gates, 4 * filters), 4, axis=3)
  new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
  return tf.sigmoid(g[2]) * tf.tanh(new_cell)



def diagonal_conv_gru(x,
                      kernel_size,
                      filters,
                      dropout=0.0,
                      name=None,
                      reuse=None):
  """Diagonal Convolutional GRU as in https://arxiv.org/abs/1702.08727."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start):
    return conv(
        args,
        filters,
        kernel_size,
        padding="SAME",
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  reset, reset_cost = hard_sigmoid(do_conv(x, "reset", 0.5))
  gate, gate_cost = hard_sigmoid(do_conv(x, "gate", 0.7))
  candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0))

  if dropout > 0.0:
    candidate = tf.nn.dropout(candidate, 1.0 - dropout)

  # Diagonal shift.
  shift_filters = filters // 3
  base_filter = ([[0, 1, 0]] * (filters - 2 * shift_filters) +
                  [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
  shift_filter = tf.constant(tf.transpose(base_filter), dtype=tf.float32)
  shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
  x_shifted = tf.nn.depthwise_conv2d(
      x, shift_filter, [1, 1, 1, 1], padding="SAME")

  # Return the gated result and cost.
  total_cost_avg = 0.5 * (reset_cost + gate_cost)
  return gate * x_shifted + (1 - gate) * candidate, total_cost_avg



def approximate_split(x, num_splits, axis=0):
  """Split approximately equally into num_splits parts.
  Args:
    x: a Tensor
    num_splits: an integer
    axis: an integer.
  Returns:
    a list of num_splits Tensors.
  """
  size = shape_list(x)[axis]
  size_splits = [tf.math.divide(size + i, num_splits) for i in range(num_splits)]
  return tf.split(x, size_splits, axis=axis)


