import pandas as pd
import os

os.makedirs("trained_result/grid_world", exist_ok=True)

try:
    dec = pd.read_csv("trained_result/dec-result.csv")
    dec.insert(2,"Methodology",["Decremental" for i in range(len(dec["Model Utility"]))],True)
    dec.to_csv("trained_result/grid_world/combined_result.csv",index=False)
    print("Successfully created combined_result.csv")
except Exception as e:
    print(f"Error: {e}")
