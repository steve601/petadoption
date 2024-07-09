import os
import sys
from source.logger import logging
from source.exception import UserException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from source.commons import evaluate_model,save_object

@dataclass # allows us to only use variable within a class
class ModelTrainingConfig:
    trained_model_file_path:str = os.path.join('elements','model.pkl') #creating a path where we will save our model
    
class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainingConfig() #creating an instance of the class above
    logging.info('Modeltraining begins...') 
    def start_of_model_training(self,input_feature_train_arr,input_feature_test_arr,output_feature_train_df,output_feature_test_df):
        try:
            X_train,y_train,X_test,y_test = (
                input_feature_train_arr,
                output_feature_train_df,
                input_feature_test_arr,
                output_feature_test_df
            )
            logging.info('Initializing different models')
            models = {
                    'Logistic Regression':LogisticRegression(),
                    'SVC':SVC(),
                    'KNeighbours':KNeighborsClassifier(),
                    'Cat Boost':CatBoostClassifier(logging_level='Silent'),
                    'XGB':XGBClassifier(),
                    'Decision Tree':DecisionTreeClassifier(),
                    'Random Forest':RandomForestClassifier(),  
                    }
            logging.info('printing model report and storing the best model')
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            best_score = max(sorted(list(model_report.values())))
            for k,v in model_report.items():
                if v == best_score:
                    best_model_name = k
                    break
                
            best_model = models[best_model_name] # accessing the values
            # let's set a condition such that the accuracy must be above a certain threshold
            if best_score < 0.70:
                raise UserException('No best model found')

            save_object(
                file_path = self.modeltrainerconfig.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            score = accuracy_score(y_test,predicted)
            
            return score
        except Exception as e:
            raise UserException(e,sys)
            
        
    