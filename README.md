
## py_attention
This module contains varying attention layer implementations for use with deep learning models.

Common layers:
- [area_attention](https://arxiv.org/abs/1810.10126)
- [dilated_attention](https://github.com/tensorflow/tensor2tensor/blob/670ddafd233daab956d0b541c6bb360a86a45c1b/tensor2tensor/layers/common_attention.py#L3315)
- [dot_product_attention](https://github.com/tensorflow/tensor2tensor/blob/670ddafd233daab956d0b541c6bb360a86a45c1b/tensor2tensor/layers/common_attention.py#L1602)
- [multihead_attention](https://arxiv.org/abs/1706.03762)
- [local_attention](https://github.com/tensorflow/tensor2tensor/blob/670ddafd233daab956d0b541c6bb360a86a45c1b/tensor2tensor/layers/common_attention.py#L3191)
- [relative_attention](https://github.com/tensorflow/tensor2tensor/blob/670ddafd233daab956d0b541c6bb360a86a45c1b/tensor2tensor/layers/common_attention.py#L1739)


## Usage
To use the module in python:
```python

from attention.dot_product_attention import py_dot_product_attention
from attention.attention_utils import py_compute_qkv

import tensorflow as tf

x = tf.random.normal([8, 256, 16])

q, k, v = py_compute_qkv(x, None, 64, 64)

attention = py_dot_product_attention(q, k, v)


from attention.multihead_attention import py_multihead_attention 

x = tf.random.normal([4, 128, 8])

attention = py_multihead_attention(x, None, 64, 64, 128)

# Usually used before model caps, e.g.:
pool = tf.keras.layers.GlobalMaxPool1D()(attention)
output = tf.keras.layers.Dense(units=10)(pool)

```


## Note
The [tensor2tensor](https://github.com/tensorflow/tensor2tensor) framework is
the basis for the majority of the code contained in this package.

This [commit](https://github.com/tensorflow/tensor2tensor/tree/670ddafd233daab956d0b541c6bb360a86a45c1b) is the last commit referenced.
