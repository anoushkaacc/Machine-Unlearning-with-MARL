import pandas as pd

dec = pd.read_csv("trained_result/aircraft_landing/dec-result.csv")
dec.insert(2,"Methodology",["Decremental" for i in range(len(dec["Model Utility"]))],True)

# poi = pd.read_csv("trained_result/aircraft_landing/poi-result.csv")
# poi.insert(2,"Methodology",["Poisoning" for i in range(len(poi["Model Utility"]))],True)

res = dec
# res = pd.concat([dec,poi])
res.to_csv("trained_result/combined_result.csv",index=False)
