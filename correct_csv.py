import sys
import pandas as pd

filename = sys.argv[1]
data = pd.read_csv(f'./{filename}')
# drop the first and last column
data = data.drop([data.columns[len(data.columns)-1]], axis=1)
data = data.drop([data.columns[0]], axis=1)

# add header
sensor_name = filename.split('_')[1].split('.csv')[0]
columns = ['timestamp']
for i in range(0,len(data.columns)-1):
    columns.append(f'{sensor_name}_{i}')

data.columns = columns

# save the corrected csv file
data.to_csv(f'./_{filename}', index=False)