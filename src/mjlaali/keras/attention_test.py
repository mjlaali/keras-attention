import numpy as np
import tensorflow as tf
from tensorflow import keras

from mjlaali.keras.attention import AttentionCell
from mjlaali.tensorflow.test_utils import run_test_in_eager_mode


def test_initial_attention_shape():
    units = 10
    batch_size = 16
    tf_dtype = 'float32'

    gru_cell = keras.layers.GRUCell(units)
    attention_cell = AttentionCell(gru_cell)
    np.testing.assert_equal(
        gru_cell.get_initial_state(batch_size=batch_size, dtype=tf_dtype).shape,
        attention_cell.get_initial_state(batch_size=batch_size, dtype=tf_dtype).shape)


def test_attention_cell_output_shape():
    inputs_dim = keys_dim = units = 10
    context_length = 8
    seq_len = 4
    tf_dtype = 'float32'

    gru_cell = keras.layers.GRUCell(units)
    attention_cell = AttentionCell(gru_cell)
    rnn_layer = keras.layers.RNN(attention_cell)

    inputs = keras.layers.Input(shape=(seq_len, inputs_dim), name='seq_inputs')
    constants = keras.layers.Input(shape=(context_length, keys_dim), name='attention_inputs')
    outputs = rnn_layer(inputs, constants=constants)
    model = keras.Model(inputs=[inputs, constants], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')