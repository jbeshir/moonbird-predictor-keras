import pandas as pd

# Load in all response data.
AllData = pd.read_csv('responsedata.csv', sep=',', header=None)

UniquePredictions = AllData[0].unique()

print(AllData[0].unique())