#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from utils import load_model

def make_prediction(input_data):
    # Load the trained model
    model = load_model("rf_model.joblib")
    if model is None:
        raise FileNotFoundError("Trained model not found!")

    # Predict the class
    prediction = model.predict([input_data])
    return prediction

if __name__ == "__main__":
    # Example input data (must match feature size of Iris dataset)
    input_data = np.array([5.1, 3.5, 1.4, 0.2])
    prediction = make_prediction(input_data)
    print(f"Input: {input_data}")
    print(f"Prediction: {prediction}")

