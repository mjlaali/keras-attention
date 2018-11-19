from tensorflow import keras
from tensorflow.python.keras import backend as K


class AttentionCell(keras.layers.Layer):

    def __init__(self, rnn_cell, **kwargs):
        super().__init__(**kwargs)
        self.rnn_cell = rnn_cell
        self.query_transformation = None
        self.attention_logits_dense = None
        self.internal_layers = None

    def build(self, input_shape):
        cell_input_shape, constant_input_shape = input_shape
        self.rnn_cell.build(cell_input_shape)
        keys_dim = constant_input_shape[-1]

        self.query_transformation = keras.layers.Dense(units=keys_dim, input_shape=(self.rnn_cell.state_size,))
        self.attention_logits_dense = keras.layers.Dense(units=1, input_shape=(keys_dim,))
        self.internal_layers = [self.rnn_cell, self.query_transformation, self.attention_logits_dense]

    def call(self, inputs, states, constants):
        if not isinstance(constants, (list, tuple)):
            keys = values = constants
        elif len(constants) == 1:
            keys = values = constants[0]
        elif len(constants) == 2:
            keys, values = constants
        else:
            raise ValueError('constants can either be a list with keys and values or just attention vectors')

        if not isinstance(states, (list, tuple)):
            query = states
        else:
            query = states[0]

        query = self.query_transformation(query)
        repeated_query = K.repeat(query, K.shape(keys)[1])

        logits = self.attention_logits_dense(K.tanh(repeated_query + keys))
        attention_weights = keras.activations.softmax(logits, axis=1)
        attention_context = K.sum(attention_weights * values, axis=1, keepdims=False)
        inputs = inputs + attention_context
        return self.rnn_cell.call(inputs, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.rnn_cell.get_initial_state(inputs, batch_size, dtype)

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    @property
    def trainable_weights(self):
        weights = super(AttentionCell, self).trainable_weights
        for l in self.internal_layers:
            weights += l.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = super(AttentionCell, self).non_trainable_weights
        for l in self.internal_layers:
            weights += l.non_trainable_weights
        return weights

