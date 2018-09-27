import numpy as np
import pandas as pd

# Load in all response summary data.
AllSummaryData = pd.read_csv('summarydata.csv', sep=',', header=None)

AllShuffledSummaryData = AllSummaryData[AllSummaryData[5] != 0].sample(frac=1)

n_test = np.floor(AllShuffledSummaryData.shape[0] * 0.2).astype(np.int64)
n_cv = n_test
n_train = AllShuffledSummaryData.shape[0] - n_cv - n_test

TrainSummaryData = AllShuffledSummaryData.iloc[0:n_train, :]
CVSummaryData = AllShuffledSummaryData.iloc[n_train:n_train+n_cv, :]
TestSummaryData = AllShuffledSummaryData.iloc[n_train+n_cv:, :]

TrainSummaryData.to_csv('summarydata-train.csv', index=False, header=False)
CVSummaryData.to_csv('summarydata-cv.csv', index=False, header=False)
TestSummaryData.to_csv('summarydata-test.csv', index=False, header=False)

print(n_train, n_cv, n_test)
