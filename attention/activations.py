import tensorflow as tf
from .tensor_utils import shape_list, to_float
from numpy import pi


def brelu(x):
  """Bipolar ReLU as in https://arxiv.org/abs/1709.04054."""
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.relu(x1)
  y2 = -tf.nn.relu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def belu(x):
  """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
  x_shape = shape_list(x)
  x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
  y1 = tf.nn.elu(x1)
  y2 = -tf.nn.elu(-x2)
  return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


def lrelu(input_, leak=0.2, name="lrelu"):
  return tf.math.maximum(input_, leak * input_, name=name)
  

def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    x with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (tf.math.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def nac(x, depth, name=None, reuse=None):
  """NAC as in https://arxiv.org/abs/1808.00508."""
  x_shape = shape_list(x)
  w = tf.Variable(tf.random.normal(shape=[x_shape[-1], depth]), name="w")
  m = tf.Variable(tf.random.normal(shape=[x_shape[-1], depth]), name="m")
  w = tf.tanh(w) * tf.nn.sigmoid(m)
  x_flat = tf.reshape(x, [-1, x_shape[-1]])
  res_flat = tf.matmul(x_flat, w)
  return tf.reshape(res_flat, x_shape[:-1] + [depth])


def nalu(x, depth, epsilon=1e-30, name=None, reuse=None):
  """NALU as in https://arxiv.org/abs/1808.00508."""
  x_shape = shape_list(x)
  x_flat = tf.reshape(x, [-1, x_shape[-1]])
  gw = tf.Variable(tf.random.normal(shape=[x_shape[-1], depth]), name="w")
  g = tf.nn.sigmoid(tf.matmul(x_flat, gw))
  g = tf.reshape(g, x_shape[:-1] + [depth])
  a = nac(x, depth, name="nac_lin")
  log_x = tf.math.log(tf.abs(x) + epsilon)
  m = nac(log_x, depth, name="nac_log")
  return g * a + (1 - g) * tf.exp(m)



def saturating_sigmoid(x):
  """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
  with tf.name_scope("saturating_sigmoid", values=[x]):
    y = tf.sigmoid(x)
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))


def hard_sigmoid(x, saturation_limit=0.9):
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  x_shifted = 0.5 * x + 0.5
  return tf.minimum(1.0, tf.nn.relu(x_shifted)), saturation_cost


def hard_tanh(x, saturation_limit=0.9):
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  return tf.minimum(1.0, tf.maximum(x, -1.0)), saturation_cost


def inverse_exp_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
  inv_base = tf.exp(tf.math.log(min_value) / float(max_step))
  if step is None:
    step = tf.compat.v1.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)
  return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
  if step is None:
    step = tf.compat.v1.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)
  progress = tf.minimum(step / float(max_step), 1.0)
  return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
  """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
  if step is None:
    step = tf.compat.v1.train.get_global_step()
  if step is None:
    return 1.0
  step = to_float(step)

  def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

  def inv_sigmoid(y):
    return tf.math.log(y / (1 - y))

  assert min_value > 0, (
      "sigmoid's output is always >0 and <1. min_value must respect "
      "these bounds for interpolation to work.")
  assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

  # Find
  #   x  s.t. sigmoid(x ) = y_min and
  #   x' s.t. sigmoid(x') = y_max
  # We will map [0, max_step] to [x_min, x_max].
  y_min = min_value
  y_max = 1.0 - min_value
  x_min = inv_sigmoid(y_min)
  x_max = inv_sigmoid(y_max)

  x = tf.minimum(step / float(max_step), 1.0)  # [0, 1]
  x = x_min + (x_max - x_min) * x  # [x_min, x_max]
  y = sigmoid(x)  # [y_min, y_max]

  y = (y - y_min) / (y_max - y_min)  # [0, 1]
  y = y * (1.0 - y_min)  # [0, 1-y_min]
  y += y_min  # [y_min, 1]
  return y

