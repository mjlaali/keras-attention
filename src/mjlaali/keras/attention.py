from tensorflow import keras
from tensorflow.python.keras import backend as K


class AttentionCell(keras.layers.Layer):

    def __init__(self, rnn_cell, **kwargs):
        super(AttentionCell, self).__init__(**kwargs)
        self._cell = rnn_cell
        self._query_transformation = None
        self._attention_logits_dense = None
        self._internal_layers = None

    def build(self, input_shape):
        super(AttentionCell, self).build(input_shape)
        cell_input_shape, constant_input_shape = input_shape
        self._cell.build(cell_input_shape)
        keys_dim = constant_input_shape[-1]

        self._query_transformation = keras.layers.Dense(units=keys_dim, input_shape=(self._cell.state_size,))
        self._attention_logits_dense = keras.layers.Dense(units=1, input_shape=(keys_dim,))
        self._internal_layers = [self._cell, self._query_transformation, self._attention_logits_dense]

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

        query = self._query_transformation(query)
        repeated_query = K.repeat(query, K.shape(keys)[1])

        logits = self._attention_logits_dense(K.tanh(repeated_query + keys))
        attention_weights = keras.activations.softmax(logits, axis=1)
        attention_context = K.sum(attention_weights * values, axis=1, keepdims=False)
        inputs = inputs + attention_context
        return self._cell.call(inputs, states)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self._cell.get_initial_state(inputs, batch_size, dtype)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def trainable_weights(self):
        weights = list(super(AttentionCell, self).trainable_weights)
        for l in self._internal_layers:
            weights += l.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = list(super(AttentionCell, self).non_trainable_weights)
        for l in self._internal_layers:
            weights += l.non_trainable_weights
        return weights

    def get_config(self):
        base_config = dict(super(AttentionCell, self).get_config())
        cell_config = self._cell.get_config()
        base_config['cell'] = {
            'class_name': self._cell.__class__.__name__,
            'config': cell_config
        }
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
