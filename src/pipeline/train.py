# Corrected import statements in training_pipeline.py
import os
import sys
# import logging
import sys
import sys


from src.component.data_transformation import DataTransformation
from src.component.model_trainer import Model_trainer

from logg import logging

from src.component.dt_ing import DataIngestion
from exce import error_message_detail
import pandas as pd
 
if __name__ == '__main__':
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_class_ingestion()
        print(train_data_path, test_data_path)
        
        data_transformation=DataTransformation()
        
        x_train,x_test,y_train,y_test,obj_path=DataTransformation().initiate_data_transform(train_data_path,test_data_path)
        model_trainer= Model_trainer()
        
        model_trainer.initiate_model_trainer(x_train,x_test,y_train,y_test)
      
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(e)
# python src1/pipeline/training_pipeline.py