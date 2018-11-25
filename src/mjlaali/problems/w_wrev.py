import numpy as np
import os, shutil
import collections
from tensorflow import keras

from mjlaali.keras.attention import AttentionCell


def make_dataset(l, dataset_size, num_token):
    x = np.random.randint(low=0, high=num_token, size=(dataset_size, l))
    rev_idx = list(reversed(range(l)))
    y = x[:, rev_idx]
    y_onehot = keras.utils.to_categorical(y, num_classes=num_token)
    return x, y_onehot


def make_lstm_model(num_token, embedding_dim, units, num_layers):
    x = keras.Input(shape=(None,), dtype='int32')
    embeddings = keras.layers.Embedding(input_dim=num_token, output_dim=embedding_dim)(x)

    lstm_outputs = embeddings
    for _ in range(num_layers):
        lstm_outputs = keras.layers.Bidirectional(
            keras.layers.LSTM(units=units, return_sequences=True))(lstm_outputs)

    y = keras.layers.Dense(units=num_token, activation='softmax')(lstm_outputs)
    return keras.models.Model(inputs=x, outputs=y)


def make_attention_model(num_token, embedding_dim, units, num_layers):
    x = keras.Input(shape=(None,), dtype='int32')
    embeddings = keras.layers.Embedding(input_dim=num_token, output_dim=embedding_dim)(x)

    lstm_outputs = embeddings

    lstm_outputs = keras.layers.Bidirectional(
        keras.layers.LSTM(units=units, return_sequences=True))(lstm_outputs)

    gru_cell = keras.layers.GRUCell(units)
    attention_cell = AttentionCell(gru_cell)
    rnn_layer = keras.layers.RNN(attention_cell, return_sequences=True)

    lstm_outputs = rnn_layer(lstm_outputs, constants=lstm_outputs)

    y = keras.layers.Dense(units=num_token, activation='softmax')(lstm_outputs)
    return keras.models.Model(inputs=x, outputs=y)


def experiment(model_factory, name, epochs=10):
    embedding_dim = 10
    lstm_units = 10
    num_layers = 2
    num_token = 10

    model = model_factory(num_token=num_token, embedding_dim=embedding_dim, units=lstm_units, num_layers=num_layers)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    num_token = 10
    training_size = int(1e4)
    batch_size = 10

    for length in [10, 15, 20]:
        base_dir = 'ckpts/%s-%d' % (name, length)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        callbacks = [
            keras.callbacks.EarlyStopping(patience=100),
            keras.callbacks.ModelCheckpoint(filepath='%s/ckpt' % base_dir),
            keras.callbacks.TensorBoard(log_dir='%s/logs' % base_dir)
        ]

        x, y_onehot = make_dataset(l=length, dataset_size=training_size, num_token=num_token)
        # x_valid, y_valid, y_valid_onehot = make_dataset(l=length, dataset_size=validation_size, num_token=num_token)

        history = model.fit(
            x=x,
            y=y_onehot,
            batch_size=batch_size,
            validation_split=0.2,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks
        )

        x_test, y_onehot_test = make_dataset(l=length, dataset_size=4, num_token=num_token)
        y_hat = model.predict(x_test)
        print("len = %d, eval_acc=%s" % (length, history.history['val_acc'][-1]))
        print("x = \n%s\n\nexpected=\n%s\n\npredictions\n%s" % (
            x_test, np.argmax(y_onehot_test, axis=-1), np.argmax(y_hat, axis=-1)))


if __name__ == '__main__':
    ckpt_dir = 'ckpts'
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.mkdir(ckpt_dir)

    epochs = 1000
    experiment(make_attention_model, name='att', epochs=epochs)
    print("=============")
    experiment(make_lstm_model, name='lstm', epochs=epochs)
