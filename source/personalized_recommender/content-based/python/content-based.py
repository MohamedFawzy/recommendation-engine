import pandas as pd
import numpy as np
import scipy
import sklearn

path = "anonymous-msweb.test"

raw_data = pd.read_csv(path,header=None,skiprows=7)
print(raw_data.head())
