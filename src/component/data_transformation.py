from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utills import save_object  # Assuming you have a save_object function in utils module
import numpy as np
import pandas as pd
from logg import logging
import os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')



# # Assuming you have defined logging and DataTransformationConfig classes

# # Define the DataTransformation class
# class DataTransformation:
#     def __init__(self):
#         self.datatransformationconfig = DataTransformationConfig()

#     def get_data_transformation_obj(self):
#         try:
#             # Define pipelines for numerical and categorical features
#             num_pipeline = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='median')),
#                 ('scaler', StandardScaler())
#             ])

#             cat_pipeline = Pipeline(steps=[
#                 ('imputer', SimpleImputer(strategy='most_frequent')),
#                 ('ordinal_encoder', OrdinalEncoder()),
#                 ('scaler', StandardScaler())
#             ])

#             # Define ColumnTransformer to apply different pipelines to different columns
#             preprocessor = ColumnTransformer(transformers=[
#                 ('num', num_pipeline, numerical_cols),
#                 ('cat', cat_pipeline, categorical_cols)
#             ])

#             return preprocessor
#         except Exception as e:
#             logging.error(f'Exception occurred in data transformation: {e}')

#     def initiate_data_transform(self, train_data_path, test_data_path):
#         try:
#             # Read data from CSV files
#             train_df = pd.read_csv(train_data_path)
#             test_df = pd.read_csv(test_data_path)

#             # Define columns
#             categorical_cols = ['cut', 'color', 'clarity']
#             numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
#             target_column = 'price'

#             # Drop unwanted columns
#             train_df = train_df.drop(columns=['Unnamed: 0'], axis=1)
#             test_df = test_df.drop(columns=['Unnamed: 0'], axis=1)

#             # Get input features and target
#             X_train = train_df.drop(columns=[target_column])
#             y_train = train_df[target_column]
#             X_test = test_df.drop(columns=[target_column])
#             y_test = test_df[target_column]
            
#             # Get preprocessor object
#             preprocessor_obj = self.get_data_transformation_obj()

#             # Fit-transform on train data and transform on test data
#             X_train_transformed = preprocessor_obj.fit_transform(X_train)
            
#             X_test_transformed = preprocessor_obj.transform(X_test)
#             print('hi')
#             # Save preprocessor object
#             save_object(self.datatransformationconfig.preprocessor_obj_file_path, preprocessor_obj)

#             return X_train_transformed, y_train, X_test_transformed, y_test
#         except Exception as e:
#             logging.error(f'Exception occurred during data transformation: {e}')


import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from utills import save_object  # Assuming you have a save_object function
# from DataTransformationConfig import DataTransformationConfig  # Import your DataTransformationConfig class

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def get_data_transformation_obj(self, numerical_cols, categorical_cols):
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder()),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            return preprocessor
        except Exception as e:
            logging.error(f'Exception occurred in data transformation: {e}')

    def initiate_data_transform(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            target_column = 'price'

            train_df = train_df.drop(columns=['Unnamed: 0'], axis=1)
            test_df = test_df.drop(columns=['Unnamed: 0'], axis=1)

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]
            
            preprocessor_obj = self.get_data_transformation_obj(numerical_cols, categorical_cols)

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)
            print('hi')
            save_object(self.datatransformationconfig.preprocessor_obj_file_path, preprocessor_obj)
          
            return X_train_transformed, X_test_transformed,y_train,y_test,self.datatransformationconfig.preprocessor_obj_file_path
        except Exception as e:
            logging.error(f'Exception occurred during data transformation: {e}')
            # Handle the exception appropriately, such as raising or returning an error message

# Example usage
# if __name__ == "__main__":
#     data_transformer = DataTransformation()
#     X_train_trans, y_train, X_test_trans, y_test = data_transformer.initiate_data_transform('train.csv', 'test.csv')
#     print(X_train_trans.shape, y_train.shape, X_test_trans.shape, y_test.shape)
# x=DataTransformation()
# x.initiate_data_transform(r"C:\Users\balkr\OneDrive\Desktop\pp\artifacts\train.csv",r"C:\Users\balkr\OneDrive\Desktop\pp\artifacts\test.csv")

# git add .hjhj