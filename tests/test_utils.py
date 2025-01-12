#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from sklearn.ensemble import RandomForestClassifier
from src.utils import save_model, load_model

def test_save_and_load_model():
    file_path = "test_rf_model.joblib"
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Test saving the model
    save_model(model, file_path)
    assert os.path.exists(file_path), "Model file not saved!"
    
    # Test loading the model
    loaded_model = load_model(file_path)
    assert loaded_model is not None, "Model not loaded!"
    assert isinstance(loaded_model, RandomForestClassifier), "Loaded model type is incorrect!"
    
    # Clean up
    os.remove(file_path)

if __name__ == "__main__":
    test_save_and_load_model()
    print("All tests for utils passed!")

