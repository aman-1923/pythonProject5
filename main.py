import pickle
import numpy as np

# Load the model from the pickle file
with open('model (1).pkl', 'rb') as f:
    model = pickle.load(f)

# Create some input data to make predictions on
input_data = np.array([[22, 15, 5, 5, 4, 4, 1]])

# Use the model to make predictions
predictions = model.predict(input_data)
print(predictions)
