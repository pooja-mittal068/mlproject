import os
import sys

<<<<<<< HEAD
import numpy as np 
import pandas as pd
import dill
import pickle
=======
import numpy as np
import pandas as pd
import dill
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
<<<<<<< HEAD
=======
from src.logger import logging
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
<<<<<<< HEAD
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
=======
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

<<<<<<< HEAD
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model
=======
            gs = GridSearchCV(model,para,cv=3)         # Apply GridSearch CV
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)      # Train Model
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

<<<<<<< HEAD
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
=======

    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
>>>>>>> a919872dc5dc7e47a4f4652a2ab672308f8ee68d
