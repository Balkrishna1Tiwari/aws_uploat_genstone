
import numpy as np 

import pandas as pd

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from logg import logging
from exce import Custom_Exception
from dataclasses import dataclass

from utills import load_object
        
        
import os
import sys
import logging
import pickle  # Missing import for loading objects
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Assuming these are custom modules in your project
from logg import logging  # Correct the import path as per your project structure
from exce import Custom_Exception  # Correct the import path as per your project structure



class Prediction_pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('Exception occurred at prediction pipeline')
            raise Custom_Exception(str(e), sys)  # Assuming Custom_Exception handles exceptions correctly


@dataclass
class CustomData:
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str

    def get_data_as_data_frame(self):
        try:
            custom_data_df = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_df)
            logging.info('Data frame gathered')
            return df
        except Exception as e:
            logging.info('Error occurred during data frame creation')
            raise Custom_Exception(str(e), sys)  # Assuming Custom_Exception handles exceptions correctly


# Example usage:
if __name__ == "__main__":
    custom_data = CustomData(carat=0.5, depth=60.0, table=55.0, x=5.1, y=5.2, z=3.1,
                             cut='Ideal', color='E', clarity='SI1')
    features_df = custom_data.get_data_as_data_frame()

    pipeline = Prediction_pipeline()
    prediction = pipeline.predict(features_df)
    print("Prediction:", prediction)
