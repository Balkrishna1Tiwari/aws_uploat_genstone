import numpy as np 

import pandas as pd

from sklearn.linear_model import LinearRegression,Lasso,ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from logg import logging
from exce import Custom_Exception
from dataclasses import dataclass
import os,sys

from utills import evaluate_model,save_object

@dataclass

class Model_trainer_config:
    
    model_trainer_config=os.path.join('artifacts','model.pkl')
    
class Model_trainer:
    def __init__(self):
        
        self.model_trainer_config1=Model_trainer_config()
        
    def initiate_model_trainer(self,x_train,x_test,y_train,y_test):
        
        logging.info('splitting dependent and independ variables from train and test')
        
        # X_train,y_train,X_test,y_test=(train_array[::-1],
        #                                train_array[:-1],
        #                                test_array[::-1],
        #                                test_array[:-1])
        # print(f"x_train shape: {x_train}, y_train shape: {y_train}")
        # print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


        models={'LinearRegression':LinearRegression(),
                'Lasso':Lasso(alpha=0.1,max_iter=10000),
                
                'elasticnet':ElasticNet(),
                'decesionTree':DecisionTreeRegressor(),
                'Random_forest':RandomForestRegressor()}
        model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
        
        logging.info(f'model report {model_report}')
        
        # best model score
        
        best_model_score=max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        logging.info(f"best model found ,model_name:{best_model_name}")
        logging.info(f"best model found ,model_score:{best_model_score}")  
        best_model=models[best_model_name]
        save_object(
            
            self.model_trainer_config1.model_trainer_config,best_model)
            
            
            

            
        
            
        
        

