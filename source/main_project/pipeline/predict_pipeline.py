import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging

class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        preprocessor_path = 'elements\preprocessor.pkl'
        # loaeding objects
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        data_preprocessed = preprocessor.transform(features)
        prediction = model.predict(data_preprocessed)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,pettype,breed,agemonths,color,size,weightkg,
                vaccinated,healthcondition,adoptionfee,previousowner):
        
        self.pet = pettype
        self.breed = breed
        self.age = agemonths
        self.color = color
        self.siz = size
        self.kg = weightkg
        self.vac = vaccinated
        self.hth = healthcondition
        self.fee = adoptionfee
        self.own = previousowner
    logging.info("Converting user's input to df")  
    # let's write a function that returns the user input as a pandas dataframe
    def get_data_as_df(self):
        try:
            columns = ['pettype', 'breed', 'agemonths', 'color', 'size', 'weightkg',
                       'vaccinated', 'healthcondition', 'adoptionfee', 'previousowner']
            user_data = [[self.pet,self.breed,self.age,self.color,self.siz,self.kg,self.vac,self.hth,self.fee,self.own]]
            return pd.DataFrame(user_data,columns=columns)
        except Exception as e:
            raise UserException(e,sys)
        