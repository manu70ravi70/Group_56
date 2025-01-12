#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib

def save_model(model, file_path):
    """Save the model to a file using Joblib."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load the model from a file using Joblib."""
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

