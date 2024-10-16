import pandas as pd
import numpy as np

# Reading file
data = pd.read_csv('../data/moscow_flats_dataset.csv')

# deleting link
data = data.drop(['link'], axis=1)

# Dropping strings where there are missing values that absent rarely
data = data.dropna(subset=['price', 'is_new', 'is_apartments', 'floor', 'number_of_floors', 'total_area', 'region_of_moscow'])

# Filling the rest of missing values with average ones
columns = ['min_to_metro', 'living_area', 'construction_year', 'ceiling_height']
for cur_col in columns:
    mean_val = data[cur_col].mean()
    data[cur_col] = data[cur_col].fillna(mean_val)

# consider only flats with prices < 100'000'000
new = data[data['price'] < 1e8]
data = new.copy(deep=True)

# adding new features
data['living_area_ratio'] = data['living_area'] / data['total_area']
data['average_room'] = data['living_area'] / data['number_of_rooms']
data['highness'] = data['floor'] / data['number_of_floors']

# getting better distribution for features
data['min_to_metro'] = np.log(data['min_to_metro'] + 1)
data['construction_year'] = np.log(data['construction_year'] + 1)
data['number_of_floors'] = np.log(data['number_of_floors'] + 1)
data['floor'] = np.log(data['floor'] + 2)
data['living_area'] = np.log(data['living_area'] + 1)
data['total_area'] = np.log(data['total_area'] + 1)

# dealing with category features
data = data.join(pd.get_dummies(data.region_of_moscow, dtype=float)).drop(['region_of_moscow'], axis=1)

data.to_csv("../prepared_data/prepared_dataset.csv")

