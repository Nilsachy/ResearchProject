import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import pad_sequences


def build_classification_model(X_train, y_train):
    # Padding
    X_padded = pad_sequences(X_train)
    print(X_padded)

    # Step 2: RNN Model Definition
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_padded.shape[2], X_padded.shape[1])))  # Transpose dimensions
    model.add(Dense(1, activation='sigmoid'))

    # Step 3: Training
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_data = np.transpose(X_padded, (0, 2, 1))  # Transpose dimensions
    model.fit(training_data, y_train, epochs=10, batch_size=1)
    return model