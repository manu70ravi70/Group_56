#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.inference import make_prediction
from src.utils import save_model

def test_make_prediction():
    # Mock model and input data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit([[5.1, 3.5, 1.4, 0.2]], [0])  # Train with minimal data for testing
    save_model(model, "rf_model.joblib")  # Match the filename expected by inference.py
    
    input_data = np.array([5.1, 3.5, 1.4, 0.2])
    prediction = make_prediction(input_data)
    assert prediction == [0], f"Prediction is incorrect! Expected [0], got {prediction}"
    
    print("Prediction test passed!")

if __name__ == "__main__":
    test_make_prediction()


