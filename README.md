# ML model predicting prices of flats in Moscow

Uses scikit RandomForestRegressor. data_prep prepares csv files from subdirectory data and saves it in prepared_data. ml-model takes features and predicts price. EDA was made for this case (in file EDA.ipynb).

## Featues 

All in double format

1) min_to_metro 
2) total_area
3) living_area
4) floor
5) number_of_floors
6) construction_year
7) is_new (1.0/0.0)
8) is_apartments (1.0/0.0)
9) ceiling_height
10) number_of_rooms
11) living_area_ratio
12) average_room
13) highness
14) ВАО (1.0/0.0)
15) ЗАО (1.0/0.0)
16) САО (1.0/0.0)
17) СВАО (1.0/0.0)
18) СЗАО (1.0/0.0)
19) ЦАО (1.0/0.0)
20) ЮАО (1.0/0.0)
21) ЮВАО (1.0/0.0)
22) ЮЗАО (1.0/0.0)

## Running

**Works only with Linux**

From model subdirectory enter commands:

    python3 data_prep.py
to convert csv file data/moscow_flats_dataset.csv to good format and save in prepared_data

    python3 ml-model.py train *.csv predict *
to train on file *.csv and predict for 22 features in *

## Example of input and output

Input:

    python3 ml-model.py train prepared_dataset.csv predict 0 31633073.0 3.2188758248682006 4.177459468932607 3.5085558999826545 2.5649493574615367 2.833213344056216 7.6128310304 1.0 0.0 3.15 2.0 0.5046728971962616 16.2 0.6875 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0

Output:

    Predicted price: 8183877.394957983 rub
    Quality of model is 82.72544673045464%
    So minimal answer is 6770149 rub; maximal answer is 9597606 rub
