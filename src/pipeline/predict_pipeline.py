import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predictpipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            features = features.copy()
            for col in ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]:
                features[col] = features[col].fillna("unknown")

            expected_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "writing_score",
                "reading_score",
            ]
            features = features.reindex(columns=expected_columns)

            data_scaled = preprocessor.transform(features)
            model_prediction = model.predict(data_scaled)
            return model_prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, writing_score: float, reading_score: float):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender or "unknown"],
                "race_ethnicity": [self.race_ethnicity or "unknown"],
                "parental_level_of_education": [self.parental_level_of_education or "unknown"],
                "lunch": [self.lunch or "unknown"],
                "test_preparation_course": [self.test_preparation_course or "unknown"],
                "writing_score": [self.writing_score if self.writing_score is not None else 0.0],
                "reading_score": [self.reading_score if self.reading_score is not None else 0.0]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)