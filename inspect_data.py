import pickle
import numpy as np

file_path = "trained_data/s3-dec-cRewards-Unlearn-epoch-10.pkl"
try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"Data Type: {type(data)}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            print(f"First 10 items: {data[:10]}")
        elif isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"First 10 items: {data[:10]}")
        else:
            print(data)
except Exception as e:
    print(f"Error: {e}")
