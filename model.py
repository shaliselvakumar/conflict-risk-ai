from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model():

    model = Sequential()

    model.add(Dense(32, activation='relu', input_shape=(2,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model