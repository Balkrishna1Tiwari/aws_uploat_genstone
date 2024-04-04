import os
import pickle
import sys
from logg import logging  # Assuming logg.py contains your logging setup
from exce import Custom_Exception  # Assuming exce.py contains your Custom_Exception class
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        error_msg = f"Error occurred while saving object: {str(e)}"
        logging.error(error_msg)
        raise Custom_Exception(error_msg)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    
    try:
        
        report={}
        
        for i in range(len(models)):
            
            model=list(models.values())[i]
            model.fit(x_train,y_train)
            # test data/model
            
            y_pred_test=model.predict(x_test)
            
            test_model_score=r2_score(y_test,y_pred_test)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report
            
            
            
    except Exception as e:
        
        logging.info('exception occureda t model training')
    
        raise Custom_Exception(e,sys)
    
import pickle
import logging  # Make sure to import logging module

def load_object(file_path:str) -> object:
    try:
        with open(file_path, 'rb') as file_obj:
            obj= pickle.load(file_obj)
        return obj
    except Exception as e:
        logging.exception('Exception occurred while loading object')
        raise Custom_Exception(str(e), sys)  # Assuming Custom_Exception handles exceptions correctly
