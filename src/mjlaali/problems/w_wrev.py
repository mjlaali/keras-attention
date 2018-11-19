import numpy as np
import collections
from tensorflow import keras


def make_dataset(l, dataset_size, num_token):
    x = np.random.randint(low=0, high=num_token, size=(dataset_size, l))
    rev_idx = list(reversed(range(l)))
    y = x[:, rev_idx]
    y_onehot = keras.utils.to_categorical(y, num_classes=num_token)
    return x, y_onehot


def make_lstm_model(num_token, embedding_dim, lstm_units, num_layers):
    x = keras.Input(shape=(None,), dtype='int32')
    embeddings = keras.layers.Embedding(input_dim=num_token, output_dim=embedding_dim)(x)

    lstm_outputs = embeddings
    for _ in range(num_layers):
        lstm_outputs = keras.layers.Bidirectional(
            keras.layers.LSTM(units=lstm_units, return_sequences=True))(lstm_outputs)

    y = keras.layers.Dense(units=num_token, activation='softmax')(lstm_outputs)
    return keras.models.Model(inputs=x, outputs=y)


def try_with_bi_lstm():
    embedding_dim = 10
    lstm_units = 20
    num_layers = 2
    num_token = 10

    model = make_lstm_model(num_token=num_token, embedding_dim=embedding_dim, lstm_units=lstm_units,
                            num_layers=num_layers)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    num_token = 10
    training_size = int(1e4)
    batch_size = 100

    for length in [2, 4, 8, 16]:
        x, y_onehot = make_dataset(l=length, dataset_size=training_size, num_token=num_token)
        # x_valid, y_valid, y_valid_onehot = make_dataset(l=length, dataset_size=validation_size, num_token=num_token)

        history = model.fit(x=x, y=y_onehot, batch_size=batch_size, validation_split=0.2, epochs=10)

        x_test, y_onehot_test = make_dataset(l=length, dataset_size=4, num_token=num_token)
        y_hat = model.predict(x_test)
        print("len = %d, eval_acc=%s" % (length, history.history['val_acc'][-1]))
        print("x = \n%s\n\nexpected=\n%s\n\npredictions\n%s" % (
        x_test, np.argmax(y_onehot_test, axis=-1), np.argmax(y_hat, axis=-1)))


if __name__=='__main__':
    try_with_bi_lstm()