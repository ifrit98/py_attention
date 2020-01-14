from .attention_utils import py_compute_qkv, py_compute_attention_component, split_heads, combine_heads
from .tensor_utils import cast_like, shape_list
from .common_layers import dense

from .dot_product_attention import py_dot_product_attention
from .dilated_attention import py_dilated_self_attention_1d, py_masked_dilated_self_attention_1d
from .local_attention import py_local_attention_1d
from .area_attention import py_dot_product_area_attention
from .relative_attention import py_dot_product_attention_relative, py_dot_product_unmasked_self_attention_relative_v2

import tensorflow as tf
from tensorflow.python.ops import inplace_ops



def py_multihead_attention(query_antecedent,
                           memory_antecedent,
                           total_key_depth,
                           total_value_depth,
                           output_depth,
                           num_heads=4,
                           dropout_rate=0,
                           bias=None,
                           attention_type="dot_product",
                           max_relative_position=None,
                           heads_share_relative_embedding=False,
                           add_relative_to_values=False,
                           image_shapes=None,
                           block_length=128,
                           block_width=128,
                           q_filter_width=1,
                           kv_filter_width=1,
                           q_padding="VALID",
                           kv_padding="VALID",
                           cache=None,
                           gap_size=0,
                           num_memory_blocks=2,
                           name="multihead_attention",
                           dropout_broadcast_dims=None,
                           vars_3d=False,
                           layer_collection=None,
                           recurrent_memory=None,
                           chunk_number=None,
                           hard_attention_k=0,
                           gumbel_noise_weight=0.0,
                           max_area_width=1,
                           max_area_height=1,
                           memory_height=1,
                           area_key_mode="mean",
                           area_value_mode="sum",
                           training=True,
                           **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0

  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")

  if recurrent_memory is not None:
    (
        recurrent_memory_transaction,
        query_antecedent, memory_antecedent, bias,
    ) = recurrent_memory.pre_attention(
        chunk_number,
        query_antecedent, memory_antecedent, bias,
    )

  if cache is None or memory_antecedent is None:
    q, k, v = py_compute_qkv(query_antecedent, memory_antecedent,
                             total_key_depth, total_value_depth, q_filter_width,
                             kv_filter_width, q_padding, kv_padding,
                             vars_3d_num_heads=vars_3d_num_heads,
                             layer_collection=layer_collection)
  if cache is not None:
    if attention_type not in ["dot_product", "dot_product_relative"]:
      # TODO(petershaw): Support caching when using relative position
      # representations, i.e. "dot_product_relative" attention.
      raise NotImplementedError(
          "Caching is not guaranteed to work with attention types other than"
          " dot_product.")
    if bias is None:
      raise ValueError("Bias required for caching. See function docstring "
                       "for details.")

    if memory_antecedent is not None:
      # Encoder-Decoder Attention Cache
      q = py_compute_attention_component(query_antecedent, total_key_depth,
                                         q_filter_width, q_padding, "q",
                                         vars_3d_num_heads=vars_3d_num_heads)
      k = cache["k_encdec"]
      v = cache["v_encdec"]
    else:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)
      decode_loop_step = kwargs.get("decode_loop_step")
      if decode_loop_step is None:
        k = cache["k"] = tf.concat([cache["k"], k], axis=2)
        v = cache["v"] = tf.concat([cache["v"], v], axis=2)
      else:
        # Inplace update is required for inference on TPU.
        # Inplace_ops only supports inplace_update on the first dimension.
        # The performance of current implementation is better than updating
        # the tensor by adding the result of matmul(one_hot,
        # update_in_current_step)
        tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
        tmp_k = inplace_ops.alias_inplace_update(
            tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
        k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
        tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
        tmp_v = inplace_ops.alias_inplace_update(
            tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
        v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

  q = split_heads(q, num_heads)
  if cache is None:
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)

  key_depth_per_head = total_key_depth // num_heads
  if not vars_3d:
    q *= key_depth_per_head**-0.5

  additional_returned_value = None
  if callable(attention_type):  # Generic way to extend multihead_attention
    x = attention_type(q, k, v, **kwargs)
    if isinstance(x, tuple):
      x, additional_returned_value = x  # Unpack
  elif attention_type == "dot_product":
    if max_area_width > 1 or max_area_height > 1:
      x = py_dot_product_area_attention(
          q, k, v, bias, dropout_rate, image_shapes,
          dropout_broadcast_dims=dropout_broadcast_dims,
          max_area_width=max_area_width,
          max_area_height=max_area_height,
          memory_height=memory_height,
          area_key_mode=area_key_mode,
          area_value_mode=area_value_mode,
          training=training)
    else:
      x = py_dot_product_attention(
          q, k, v, bias, dropout_rate, image_shapes,
          dropout_broadcast_dims=dropout_broadcast_dims,
          activation_dtype=kwargs.get("activation_dtype"),
          hard_attention_k=hard_attention_k,
          gumbel_noise_weight=gumbel_noise_weight)
  elif attention_type == "dot_product_relative":
    x = py_dot_product_attention_relative(
        q,
        k,
        v,
        bias,
        max_relative_position,
        dropout_rate,
        image_shapes,
        make_image_summary=make_image_summary,
        cache=cache is not None,
        allow_memory=recurrent_memory is not None,
        hard_attention_k=hard_attention_k,
        gumbel_noise_weight=gumbel_noise_weight)
  elif attention_type == "dot_product_unmasked_relative_v2":
    x = py_dot_product_unmasked_self_attention_relative_v2(
        q,
        k,
        v,
        bias,
        max_relative_position,
        dropout_rate,
        image_shapes,
        make_image_summary=make_image_summary,
        dropout_broadcast_dims=dropout_broadcast_dims,
        heads_share_relative_embedding=heads_share_relative_embedding,
        add_relative_to_values=add_relative_to_values)
  # # MASKED attention functions... tbd if needed to implement
  # elif attention_type == "dot_product_relative_v2":
  #   x = py_dot_product_self_attention_relative_v2(
  #       q,
  #       k,
  #       v,
  #       bias,
  #       max_relative_position,
  #       dropout_rate,
  #       image_shapes,
  #       save_weights_to=save_weights_to,
  #       make_image_summary=make_image_summary,
  #       dropout_broadcast_dims=dropout_broadcast_dims,
  #       heads_share_relative_embedding=heads_share_relative_embedding,
  #       add_relative_to_values=add_relative_to_values)
  # elif attention_type == "local_within_block_mask_right":
  #   x = py_masked_within_block_local_attention_1d(
  #       q, k, v, block_length=block_length)
  # elif attention_type == "local_relative_mask_right":
  #   x = py_masked_relative_local_attention_1d(
  #       q,
  #       k,
  #       v,
  #       block_length=block_length,
  #       make_image_summary=make_image_summary,
  #       dropout_rate=dropout_rate,
  #       heads_share_relative_embedding=heads_share_relative_embedding,
  #       add_relative_to_values=add_relative_to_values,
  #       name="masked_relative_local_attention_1d")
  # elif attention_type == "local_mask_right":
  #   x = py_masked_local_attention_1d(
  #       q,
  #       k,
  #       v,
  #       block_length=block_length,
  #       make_image_summary=make_image_summary)
  elif attention_type == "local_unmasked":
    x = py_local_attention_1d(
        q, k, v, block_length=block_length, filter_width=block_width)
  elif attention_type == "masked_dilated_1d":
    x = py_masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                            gap_size, num_memory_blocks)
  elif attention_type == "unmasked_dilated_1d":
    x = py_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                     gap_size, num_memory_blocks)
  else:
    raise ValueError("attention type %s not understood", attention_type)

  x = combine_heads(x)

  # Set last dim specifically.
  x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

  if vars_3d:
    o_var = tf.Variable(
      tf.random.normal([num_heads, total_value_depth // num_heads, output_depth]),
      name = "o")
    o_var = tf.cast(o_var, x.dtype)
    o_var = tf.reshape(o_var, [total_value_depth, output_depth])
    x = tf.tensordot(x, o_var, axes=1)
  else:
    x = dense(
        x, output_depth, use_bias=False, name="output_transform",
        layer_collection=layer_collection)

  if recurrent_memory is not None:
    x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
  if additional_returned_value is not None:
    return x, additional_returned_value
  return x
