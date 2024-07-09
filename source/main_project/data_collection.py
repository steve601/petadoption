import os
import sys
from source.exception import UserException
from source.logger import logging
from source.main_project.data_transformation import DataTransformation
from source.main_project.model_trainer import ModelTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logging.info('Creation of outputs paths')
@dataclass # allow us to only use variable within a class
class DataCollectionConfig:
    # where the outputs will be saved
    raw_data_path:str = os.path.join('elements','raw.csv')
    train_data_path:str = os.path.join('elements','train.csv')
    test_data_path:str = os.path.join('elements','test.csv')

    
class DataCollection:
    def __init__(self):
        self.collection_config = DataCollectionConfig() #instantiating the class above
    def start_collection(self):
        logging.info('Start of data ingestion method')
        try:
            data = pd.read_csv('notebook\data\clean_data.csv')
            logging.info('Data collected/read successfully')
          
            # now let's create folders for all these data
            os.makedirs(os.path.dirname(self.collection_config.raw_data_path),exist_ok=True)
            #saving the raw data
            data.to_csv(self.collection_config.raw_data_path,index=False,header=True)
            logging.info('Train Test Split starting...')
            train_data,test_data = train_test_split(data,test_size=0.25,random_state=42)
            # savng the train and test data
            train_data.to_csv(self.collection_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.collection_config.test_data_path,index=False,header=True)
            
            logging.info('Data colection and saving complete!')
            
            return(
                self.collection_config.train_data_path,
                self.collection_config.test_data_path
            )
        except Exception as e:
            raise UserException(e,sys)
        
if __name__ == "__main__":
    # creating the object for dataCellection class
    obj = DataCollection()
    # accessing method for that class
    train_path,test_path = obj.start_collection()
    
    # instantiating the datatransformer class
    transformer_obj = DataTransformation()
    # accessing its equivalent method
    input_feature_train_arr,input_feature_test_arr,output_feature_train_df,output_feature_test_df = transformer_obj.start_data_transformation(train_path,test_path)
    
    # instantiating Modeltrainer class
    model_obj = ModelTrainer()
    # accessing its equivalent method
    model_obj.start_of_model_training(input_feature_train_arr,input_feature_test_arr,output_feature_train_df,output_feature_test_df)