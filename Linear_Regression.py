import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import util.util as util
from sklearn.model_selectrion import train_test_split
from sklearn.preprocessing import StandardSclaer



# open the file as a DataFrame (here the file is a .txt related to a ETF called SQQQ)

file_path = input('please enter the path to the file: ')
df = pd.read_csv(file_path, delimiter=',')


# Lets take a look ta the data
print(df.info())		
print(df.is_null())
print(df.head(2))
print(df.describe().transpose())
util.visualization(df)






