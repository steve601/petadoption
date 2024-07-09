import os
import sys
import numpy as np
import pandas as pd
from source.exception import UserException
from source.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from source.commons import save_object
from dataclasses import dataclass

# let's create the path of what will be returned i.e preprocessor
@dataclass
class DataTransformationConfig:
    preprocessor_obj_path:str = os.path.join('elements','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()
    logging.info('Obtaining preprocessor object')    
    def get_preprocessor_obj(self):
        try:
            enc_cols = ['pettype','breed','color']
            sc_cols = ['agemonths','weightkg','vaccinated','healthcondition','adoptionfee','previousowner']

            encoder = OneHotEncoder()
            scaler = StandardScaler(with_mean=False)

            enc_pipe = Pipeline(steps=[
                ('encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])
            sc_pipe = Pipeline(steps=[
                ('scaler',StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('enc_pipe',enc_pipe,enc_cols),
                ('sc_pipe',sc_pipe,sc_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise UserException(e,sys)
        
    logging.info('Fitting and transforming the data using preprocessor')    
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            # reading the data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Successfully read trin and test set')
            logging.info('Obtaining preprocessor object')
            
            preprocessor_obj = self.get_preprocessor_obj()
            target_column = 'adoptionlikelihood'
            
            logging.info('Separating into independent and dependent features for both sets')
            
            input_feature_train_df = train_df.drop(target_column,axis=1)
            output_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(target_column,axis=1)
            output_feature_test_df = test_df[target_column]
            
            logging.info('Fitting preprocessor object on the data')
         
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            #train_arr = np.c_[input_feature_train_arr,np.array(output_feature_train_df)]
            #test_arr = np.c_[input_feature_test_arr,np.array(output_feature_test_df)]
            
            save_object(
                file_path = self.data_trans_config.preprocessor_obj_path,
                obj = preprocessor_obj)
            
            return(
                input_feature_train_arr,
                input_feature_test_arr,
                output_feature_train_df,
                output_feature_test_df
            )
            
        except Exception as e:
            raise UserException(e,sys)
            
            