import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/housing.tsv', sep='\t',header=None)

train_data, test_data = train_test_split(data, train_size=0.7, random_state=17)

train_data.to_csv('data/housing_train.tsv', sep='\t', index=False, header=None)

test_data.to_csv('data/housing_test.tsv', sep='\t', index=False, header=None)