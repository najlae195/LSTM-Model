from tensorflow import keras
from tensorflow.keras import layers

def build_lstm(input_shape, lstm_units=(64,64,32), dropout=0.2, learning_rate=5e-4):
    inp = keras.Input(shape=input_shape)
    x = inp
    for i, u in enumerate(lstm_units):
        x = layers.LSTM(u, return_sequences=(i < len(lstm_units)-1))(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model
