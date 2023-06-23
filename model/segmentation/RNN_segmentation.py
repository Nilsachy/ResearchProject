import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


def build_model(X_train, y_train):
    # Define the model architecture
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=(40, 3), activation='relu'))
    model.add(Dense(40, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Reshape the training data to match the expected input shape
    training_data = np.reshape(X_train, (X_train.shape[0], 40, 3))
    # Train the model
    model.fit(training_data, y_train, epochs=10, batch_size=1)
    return model


def test_model(model, X_test, y_test):
    # Once the model is trained, you can use it to make predictions on new data
    # Assuming your test data is stored in 'X_test' as a 3D array of shape (200, 200, 3)
    predictions = model.predict(X_test)
    return predictions

