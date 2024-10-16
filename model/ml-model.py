import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        forest = RandomForestRegressor(n_estimators=119)
        self.model = forest
        self.X_test = None
        self.Y_test = None

    def fit(self, filename):
        data = pd.read_csv('../prepared_data/' + filename)
        X, y = data.drop(['price'], axis=1), data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)

        self.X_test = X_test
        self.Y_test = y_test


    def predict(self, y: pd.DataFrame):
        return self.model.predict(y)

    def eval(self):
        return self.model.score(self.X_test, self.Y_test)

if __name__ == "__main__":
    args = sys.argv
    M = Model()
    if len(args) > 1:
        if args[1] == 'train':
            if (len(args) > 2):
                M.fit(args[2])
            else:
                print("Please enter path to dataset")
        if args[3] == "predict":
            if len(args) >= 27:
                model_argument = [[]]

                for i in range(4, 27):
                    model_argument[0].append(float(args[i]))

                ma = pd.DataFrame(model_argument)
                print(ma)
                pred = M.predict(ma)[0]
                print(f"Predicted price: {pred} rub")
                r = M.eval()
                print(f"Quality of model is {r * 100}%")
                print(f"So minimal answer is {round(r * pred)} rub; maximal answer is {round((2 - r) * pred)} rub")
            else:
                print("Not enough arguments")
    else:
        print("Please choose mode")



