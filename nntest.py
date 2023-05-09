# Import necessary libraries
import numpy as np
from tensorflow import keras
from PIL import Image

# Load the trained model from a file
model = keras.models.load_model('mnist_model.h5')

# Load and preprocess the test image
img = Image.open('zero.png').convert('L')
img = img.resize((28, 28))
img = np.array(img).reshape(1, 784) / 255.0

# Use the model to make a prediction
y_pred = model.predict(img)

# Print the predicted digit
print('Predicted digit:', np.argmax(y_pred))
