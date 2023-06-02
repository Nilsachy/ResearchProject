import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


def build_model(X_train, y_train):
    # Assuming your training data is stored in 'X_train' as a 3D array of shape (?, 200, 3)
    # where ? is the number of samples, 200 is the sequence length, and 3 is the number of features.
    # Assuming your target vector is stored in 'y_train' as a 2D array of shape (200, 1)
    batch_size = X_train.shape[0]
    # Define the model architecture
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(200, 3)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=batch_size)
    return model


def test_model(model, X_test, y_test):
    # Once the model is trained, you can use it to make predictions on new data
    # Assuming your test data is stored in 'X_test' as a 3D array of shape (200, 200, 3)
    predictions = model.predict(X_test)
    return predictions
