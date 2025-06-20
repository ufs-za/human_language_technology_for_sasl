
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Simulate dataset for demonstration: images of shape (28, 28, 1) for binary classification (A vs Not A)
num_samples = 1000
image_shape = (28, 28, 1)

# Random grayscale images
X = np.random.rand(num_samples, *image_shape)

# Random labels: 1 for 'A', 0 for 'Not A'
y = np.random.randint(0, 2, size=(num_samples,))
y = to_categorical(y, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple CNN model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=image_shape),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save the model
model_path = "letter_a_recognition_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")
