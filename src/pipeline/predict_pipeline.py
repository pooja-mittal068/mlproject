import sys
import pandas as pd
from src.exception import CustomException
<<<<<<< HEAD
from src.utils import load_object

=======
from src.utils import load_object    # To load pickle file from utils.py
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
<<<<<<< HEAD
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
=======
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
<<<<<<< HEAD

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

=======
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
<<<<<<< HEAD
            raise CustomException(e, sys)
=======
            raise CustomException(e,sys)


>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d

