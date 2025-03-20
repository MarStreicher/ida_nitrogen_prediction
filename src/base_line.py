

import numpy as np
from dataset import SpectralData
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if __name__ == "__main__":
    
    # Load data
    data = SpectralData(
        domain_list = ["UNL_Maize", "UNL_Camelina", "UNL_Sorghum", "UNL_Soybean"],
        trait_list = ["N"],
        normalization = True,
        directory_path = "data"
    ) 
    
    # Calculate the mean of the trainings set
    train_mean = np.mean(data.target_train)
    mean_vector = np.array([train_mean]*len(data.target_test))
    
    # Calculate r and r2 value
    mae = mean_absolute_error(data.target_test, mean_vector)
    mse = mean_squared_error(data.target_test, mean_vector)
    r2 = r2_score(data.target_test, mean_vector)
    
    # Print
    print(f"mae: {mae}, r2 = {r2}, mse = {mse}")
    
    
    
    